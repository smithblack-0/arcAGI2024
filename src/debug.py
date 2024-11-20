from dataclasses import dataclass, field, MISSING
from typing import Any

class FixDataclassResolutionOrder(type):
    def __new__(metacls, name, bases, namespace):
        # Collect annotations and defaults from MRO in reverse (child to parent)
        annotations = {}
        non_default_fields = []
        default_fields = []

        # Function to process fields
        def process_fields(cls_namespace):
            cls_annotations = cls_namespace.get('__annotations__', {})
            for field_name, field_type in cls_annotations.items():
                if field_name in annotations:
                    continue  # Skip if already processed
                annotations[field_name] = field_type
                field_value = cls_namespace.get(field_name, MISSING)
                if isinstance(field_value, field):
                    # Check if the field has a default value
                    if field_value.default is not MISSING or field_value.default_factory is not MISSING:
                        default_fields.append(field_name)
                    else:
                        non_default_fields.append(field_name)
                elif field_value is not MISSING:
                    # It's a normal default value
                    default_fields.append(field_name)
                else:
                    # No default value
                    non_default_fields.append(field_name)

        # Start processing from the child class
        process_fields(namespace)

        # Process base classes in reverse order
        for base in reversed(bases):
            if hasattr(base, '__annotations__'):
                base_namespace = vars(base)
                process_fields(base_namespace)

        # Reconstruct __annotations__ in the desired order
        new_annotations = {}
        for field_name in non_default_fields + default_fields:
            new_annotations[field_name] = annotations[field_name]
        namespace['__annotations__'] = new_annotations

        # Ensure default values are in the namespace
        for field_name in default_fields:
            for cls in [namespace] + [vars(base) for base in reversed(bases)]:
                if field_name in cls:
                    namespace[field_name] = cls[field_name]
                    break

        return super().__new__(metacls, name, bases, namespace)

from dataclasses import dataclass

@dataclass
class GrandParent:
    gp_default: int = 0  # Default field

@dataclass
class Parent(GrandParent):
    p_default: int = 1  # Default field

@dataclass
class Child(Parent):
    c_nondefault: int    # Non-default field
