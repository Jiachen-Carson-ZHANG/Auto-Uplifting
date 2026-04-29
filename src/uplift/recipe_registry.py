"""Deterministic feature recipe registry for controlled supervisor expansion."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.models.uplift import (
    UpliftFeatureArtifact,
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
)
from src.uplift.features import (
    FeatureCohort,
    build_feature_table,
    compute_dataset_fingerprint,
)


class UpliftFeatureRecipeRegistry:
    """Approved feature recipe families and cached artifact lookup."""

    def __init__(self, recipes_by_family: Mapping[str, UpliftFeatureRecipeSpec]) -> None:
        self._recipes_by_family = dict(recipes_by_family)
        self._artifacts_by_recipe_id: dict[str, UpliftFeatureArtifact] = {}

    @classmethod
    def default(cls) -> "UpliftFeatureRecipeRegistry":
        """Return the default approved recipe families for M5."""
        return cls(
            {
                "base": UpliftFeatureRecipeSpec(
                    source_tables=["clients"],
                    feature_groups=["demographic"],
                    builder_version="v1",
                ),
                "rfm": UpliftFeatureRecipeSpec(
                    source_tables=["clients", "purchases"],
                    feature_groups=["demographic", "rfm"],
                    builder_version="v1",
                ),
                "windowed": UpliftFeatureRecipeSpec(
                    source_tables=["clients", "purchases"],
                    feature_groups=["demographic", "rfm", "basket", "points"],
                    windows_days=[30],
                    builder_version="v1",
                ),
                "engagement": UpliftFeatureRecipeSpec(
                    source_tables=["clients", "purchases"],
                    feature_groups=["demographic", "points"],
                    windows_days=[30],
                    builder_version="v1",
                ),
                "product_category": UpliftFeatureRecipeSpec(
                    source_tables=["clients", "purchases", "products"],
                    feature_groups=[
                        "demographic",
                        "rfm",
                        "basket",
                        "points",
                        "product_category",
                    ],
                    windows_days=[30],
                    builder_version="v1",
                ),
                "diversity": UpliftFeatureRecipeSpec(
                    source_tables=["clients", "purchases", "products"],
                    feature_groups=[
                        "demographic",
                        "rfm",
                        "basket",
                        "points",
                        "product_category",
                        "diversity",
                    ],
                    windows_days=[30],
                    builder_version="v1",
                ),
            }
        )

    def families(self) -> list[str]:
        """Return approved family names in stable order."""
        return sorted(self._recipes_by_family)

    def recipe_for_family(self, family: str) -> UpliftFeatureRecipeSpec:
        """Return the approved recipe for one family."""
        try:
            return self._recipes_by_family[family]
        except KeyError as exc:
            raise ValueError(f"unknown feature recipe family: {family}") from exc

    def recipe_id_for_family(self, family: str) -> str:
        """Return the deterministic recipe ID for one family."""
        return self.recipe_for_family(family).feature_recipe_id

    def recipe_for_id(self, feature_recipe_id: str) -> UpliftFeatureRecipeSpec:
        """Return the approved recipe matching a deterministic recipe ID."""
        for recipe in self._recipes_by_family.values():
            if recipe.feature_recipe_id == feature_recipe_id:
                return recipe
        raise ValueError(f"unknown feature recipe id: {feature_recipe_id}")

    def register_artifact(
        self,
        artifact: UpliftFeatureArtifact,
    ) -> UpliftFeatureArtifact:
        """Record one materialized artifact by feature recipe ID."""
        self.recipe_for_id(artifact.feature_recipe_id)
        self._artifacts_by_recipe_id[artifact.feature_recipe_id] = artifact
        return artifact

    def artifact_for_recipe_id(
        self,
        feature_recipe_id: str,
    ) -> UpliftFeatureArtifact | None:
        """Return a registered artifact for a recipe ID, if one exists."""
        return self._artifacts_by_recipe_id.get(feature_recipe_id)

    def get_or_build_artifact(
        self,
        contract: UpliftProjectContract,
        *,
        family: str,
        output_dir: str | Path,
        cohort: FeatureCohort = "train",
        chunksize: int = 100_000,
    ) -> UpliftFeatureArtifact:
        """Select a cached artifact or build an approved recipe artifact."""
        recipe = self.recipe_for_family(family)
        cached = self.artifact_for_recipe_id(recipe.feature_recipe_id)
        dataset_fingerprint = compute_dataset_fingerprint(contract)
        if (
            cached is not None
            and cached.dataset_fingerprint == dataset_fingerprint
            and cached.cohort == cohort
            and Path(cached.artifact_path).exists()
        ):
            return cached
        artifact = build_feature_table(
            contract,
            recipe=recipe,
            output_dir=output_dir,
            cohort=cohort,
            chunksize=chunksize,
        )
        return self.register_artifact(artifact)
