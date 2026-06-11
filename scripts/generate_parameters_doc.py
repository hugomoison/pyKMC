import importlib.util
import sys
import os
from typing import get_args, get_origin, Union, Optional, Any, Literal
from pydantic import BaseModel, Field
import inspect

_spec = importlib.util.spec_from_file_location(
    "pykmc.config",
    os.path.join(os.path.dirname(__file__), "..", "pykmc", "config.py"),
)
_config_module = importlib.util.module_from_spec(_spec)
sys.modules["pykmc.config"] = _config_module
_spec.loader.exec_module(_config_module)
Config = _config_module.Config


def format_type(field_type: Any) -> str:
    """Formats a Python type hint into a readable string."""
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is None:
        # Handle simple types (str, int, float, bool, etc.)
        return (
            field_type.__name__ if hasattr(field_type, "__name__") else str(field_type)
        )
    elif origin is Union:
        # Handle Optional[X] and Union[X, Y]
        formatted_args = []
        for arg in args:
            if arg is not type(None):  # Exclude NoneType for Optional
                formatted_args.append(format_type(arg))
        return " or ".join(formatted_args)
    elif origin is Literal:
        # Handle Literal['a', 'b']
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    elif hasattr(origin, "__name__"):
        # Handle generic types like list, dict, tuple, etc.
        if args:  # If there are generic arguments (e.g., List[str])
            return f"{origin.__name__}[{', '.join(format_type(a) for a in args)}]"
        else:  # If it's just a generic type without specific arguments (e.g., list)
            return origin.__name__
    else:
        return str(field_type)


def is_required_pydantic_field(field: Field) -> bool:
    """Checks if a Pydantic field is mandatory."""
    return field.is_required()


def is_optional_pydantic_field(field_type: Any) -> bool:
    """Checks if a Pydantic field's type annotation indicates it's Optional."""
    origin = get_origin(field_type)
    return origin is Union and type(None) in get_args(field_type)


def document_model(
    model_class: type[BaseModel], section_name: str, is_top_level_optional: bool
) -> str:
    """
    Generates Markdown documentation for a Pydantic BaseModel section.

    Parameters
    ----------
    model_class : type[BaseModel]
        The Pydantic BaseModel class to document.
    section_name : str
        The name of the section (e.g., 'control', 'lammps').
    is_top_level_optional : bool
        True if this entire section (BaseModel) is optional in its parent Config.

    Returns
    -------
    str
        A Markdown formatted string for the section.
    """
    lines = []

    # Determine if the section itself is mandatory or optional
    section_status = "optional" if is_top_level_optional else "mandatory"

    # Section Heading
    lines.append(f"## `{section_name.capitalize()}` Section ({section_status})\n")

    # Add the class's own docstring as a description for the section
    class_doc = inspect.getdoc(model_class)
    if class_doc:
        class_doc_indented = "\n  ".join(
            class_doc.splitlines()
        )  # Indent for details block
        lines.append(
            f"<details><summary>Section Overview</summary>\n  {class_doc_indented}\n</details>\n"
        )
    else:
        lines.append("No overview description provided for this section.\n")

    for name, field in model_class.model_fields.items():
        typ = format_type(field.annotation)

        # Determine status (mandatory, optional with default, optional without default)
        status_info = ""
        if is_required_pydantic_field(field):
            status_info = "mandatory"
        elif is_optional_pydantic_field(field.annotation) and field.default is None:
            status_info = "optional"
        elif field.default is not None:
            status_info = f"default = `{field.default!r}`"
        else:  # Covers cases like Optional[str] but with a default_factory
            status_info = "optional (default provided)"  # More explicit if default_factory is used

        # Parameter line (bullet point)
        lines.append(f"- **`{name}`** : `{typ}`, {status_info}")

        # Description as a clickable <details> block
        desc = field.description or "No description provided."
        # Indent the description for Markdown details block
        desc_indented = "\n  ".join(desc.splitlines())
        lines.append(
            f"  <details><summary>Description</summary>\n  {desc_indented}\n  </details>"
        )

    lines.append("\n---\n")  # Add a separator at the end of each section
    return "\n".join(lines)


def generate_parameters_md(
    config_class: type[BaseModel],
    output_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "docs",
        "parameters_details.md",
    ),
):
    """
    Generates a Markdown file detailing all configuration parameters from a Pydantic Config model.

    Parameters
    ----------
    config_class : type[BaseModel]
        The root Pydantic BaseModel class (e.g., `Config`) to document.
    output_file : str
        The full path to the output Markdown file. Defaults to `docs/parameters_details.md`.
    """
    lines = []

    for name, field in config_class.model_fields.items():
        field_type = field.annotation

        # Determine if the top-level section (like 'lammps') is Optional
        is_top_level_optional = is_optional_pydantic_field(field_type)

        # Get the actual BaseModel class if it's Optional[BaseModel] or direct BaseModel
        sub_model_class = None
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            sub_model_class = field_type
        elif is_top_level_optional:
            # Find the BaseModel class among the Union args (excluding NoneType)
            sub_model_class = next(
                (
                    arg
                    for arg in get_args(field_type)
                    if inspect.isclass(arg) and issubclass(arg, BaseModel)
                ),
                None,
            )

        if sub_model_class:
            lines.append(document_model(sub_model_class, name, is_top_level_optional))
        # If your top-level Config also has direct fields (not sub-models), you'd handle them here.
        # For example, if Config had a `version: str` field directly.
        # Otherwise, pass.
        else:
            pass  # No direct fields in Config that aren't sub-models to document in this style.

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    generate_parameters_md(Config)
