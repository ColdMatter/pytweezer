"""Recipes: named run settings for one experiment, shared through the Experiment Manager.

A recipe is a :class:`~pytweezer.experiment.task.TaskRequest` with a name.
:class:`RecipeBook` holds the rules for saving, finding and deleting them;
:class:`RecipeStore` persists the book as JSON. :meth:`Recipe.replay` turns a
recipe into a request, and :func:`unknown_arguments` checks a request against
an experiment's schema.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from pytweezer.experiment.queue import ModelStore, now
from pytweezer.experiment.task import TaskRequest


class RecipeError(ValueError):
    """A recipe request that can't be carried out."""


class Recipe(TaskRequest):
    """Saved run settings; ``submitter`` is whoever saved it."""

    name: str
    saved_at: datetime = Field(default_factory=now)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("a recipe needs a name")
        return name

    @property
    def recipe_key(self) -> tuple[str, str, str]:
        return (self.experiment, self.class_name, self.name)

    def replay(
        self,
        args: dict[str, Any] | None = None,
        *,
        priority: int | None = None,
        label: str | None = None,
        submitter: str = "",
    ) -> TaskRequest:
        """A request to run this recipe, ``args`` replacing its fixed arguments."""
        args = args or {}
        scanned = sorted(set(args) & {axis.argument for axis in self.scan.axes})
        if scanned:
            raise RecipeError(
                f"recipe {self.name!r} scans {scanned}; an override can't fix "
                "a scanned argument"
            )
        return TaskRequest(
            experiment=self.experiment,
            class_name=self.class_name,
            args={**self.args, **args},
            scan=self.scan,
            priority=self.priority if priority is None else priority,
            label=self.label if label is None else label,
            submitter=submitter,
        )


class RecipeState(BaseModel):
    schema_version: int = 1
    recipes: list[Recipe] = Field(default_factory=list)


class RecipeBook:
    """Every saved recipe, and the rules for changing them."""

    def __init__(self, state: RecipeState | None = None) -> None:
        self.state = state or RecipeState()

    def get(self, experiment: str, class_name: str, name: str) -> Recipe:
        key = (experiment, class_name, name.strip())
        for recipe in self.state.recipes:
            if recipe.recipe_key == key:
                return recipe
        raise RecipeError(f"no recipe {key[2]!r} for {experiment}.{class_name}")

    def save(self, recipe: Recipe, overwrite: bool = False) -> Recipe:
        """Store ``recipe``, replacing one of the same name only with ``overwrite``."""
        recipe = recipe.model_copy(update={"due_time": None, "saved_at": now()})
        for index, existing in enumerate(self.state.recipes):
            if existing.recipe_key == recipe.recipe_key:
                if not overwrite:
                    raise RecipeError(
                        f"a recipe {recipe.name!r} for {recipe.experiment}."
                        f"{recipe.class_name} already exists"
                    )
                self.state.recipes[index] = recipe
                return recipe
        self.state.recipes.append(recipe)
        return recipe

    def delete(self, experiment: str, class_name: str, name: str) -> None:
        self.state.recipes.remove(self.get(experiment, class_name, name))

    def find(
        self, experiment: str | None = None, class_name: str | None = None
    ) -> list[Recipe]:
        return sorted(
            (
                recipe
                for recipe in self.state.recipes
                if experiment in (None, recipe.experiment)
                and class_name in (None, recipe.class_name)
            ),
            key=lambda recipe: recipe.recipe_key,
        )


class RecipeStore(ModelStore):
    """Persists a :class:`RecipeState`."""

    model = RecipeState


def unknown_arguments(request: TaskRequest, schema: dict[str, Any]) -> list[str]:
    """Argument names ``request`` uses that the experiment ``schema`` doesn't declare.

    ``attribute.parameter`` names on a declared MOTMaster are left out: they
    name script parameters, which only the device can check.
    """
    motmasters = {
        attribute
        for attribute, device in schema.get("devices", {}).items()
        if device.get("motmaster") is not None
    }

    def known(name: str) -> bool:
        attribute, dot, parameter = name.partition(".")
        if dot and parameter and attribute in motmasters:
            return True
        return name in schema["arguments"]

    names = {*request.args, *(axis.argument for axis in request.scan.axes)}
    return sorted(name for name in names if not known(name))
