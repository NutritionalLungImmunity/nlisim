from typing import Any, Callable, Dict, Tuple, Type, Union

# noinspection PyPackageRequirements
from attr import attrib, attrs

# noinspection PyPackageRequirements
import numpy as np

# noinspection PyPackageRequirements
from numpy import dtype

from nlisim.cell import CellData

datatype = Union[str, dtype, Type[Any]]


# noinspection PyUnusedLocal
def name_validator(_, field_name, name: str):
    if not name.isidentifier() or not name.islower():
        raise ValueError("Invalid Name")


@attrs(kw_only=True)
class CellDataFactory:
    name: str = attrib(validator=name_validator)
    parent_class: Type[CellData] = attrib(default=CellData)
    fields: Dict[str, Tuple[datatype, int, Callable]] = attrib(factory=dict)

    def add_field(
        self,
        field_name: str,
        data_type: datatype,
        initializer,
        multiplicity: int = 1,
    ) -> 'CellDataFactory':
        if field_name in self.fields:
            raise RuntimeError(f"Trying to create the field {field_name} twice!")
        if multiplicity < 1:
            raise RuntimeError(f"Cannot create a field of multiplicity less than 1.")

        initializer_fn = None

        if initializer is None:
            # use a default
            if np.issubdtype(data_type, np.number):
                if multiplicity > 1:

                    def initializer_fn():
                        return np.zeros(shape=multiplicity, dtype=data_type)

                else:
                    if np.issubdtype(data_type, np.inexact):

                        def initializer_fn():
                            return 0.0

                    elif np.issubdtype(data_type, np.integer):

                        def initializer_fn():
                            return 0

            elif np.issubdtype(data_type, np.bool_):
                if multiplicity > 1:

                    def initializer_fn():
                        return np.zeros(shape=multiplicity, dtype=bool)

                else:

                    def initializer_fn():
                        return False

            else:
                raise RuntimeError(
                    f"For the field {field_name} of type {data_type},"
                    " you must provide an explicit initializer"
                )
        elif callable(initializer):
            initializer_fn = initializer
        else:
            # make it callable
            def initializer_fn():
                return initializer

        self.fields[field_name] = (data_type, multiplicity, initializer_fn)

        return self

    def get_cell_data_class(self) -> Type[CellData]:

        new_fields = [
            (field_name, data_type, multiplicity) if multiplicity > 1 else (field_name, data_type)
            for field_name, (data_type, multiplicity, initializer) in self.fields.items()
        ]
        fields = self.parent_class.FIELDS + new_fields
        parent_class = self.parent_class

        module_name = 'nlisim.modules.' + self.name
        class_name = self.name.capitalize() + 'CellData'
        qual_name = module_name + '.' + class_name

        class Metaclass(type):
            def __new__(mcs, cls_name, bases, attributes):
                attributes = {
                    attr: v
                    for attr, v in attributes.items()
                    if attr not in ['__module__', '__qualname__', '__name__']
                }
                attributes['__module__'] = module_name
                attributes['__qualname__'] = qual_name
                # attrs['__name__'] = class_name
                return super(Metaclass, mcs).__new__(mcs, class_name, bases, attributes)

        class CreatedCellData(parent_class, metaclass=Metaclass):
            OWN_FIELDS = new_fields
            FIELDS = fields
            dtype = np.dtype(fields, align=True)  # type: ignore

            @classmethod
            def create_cell_tuple(cls, **kwargs) -> Tuple:
                initializer = {
                    field_name: kwargs.get(field_name, initializer())
                    for field_name, (data_type, multiplicity, initializer) in self.fields.items()
                }

                # ensure that these come in the correct order
                return parent_class.create_cell_tuple(**kwargs) + tuple(
                    [initializer[key] for key, *_ in CreatedCellData.OWN_FIELDS]
                )

        return CreatedCellData
