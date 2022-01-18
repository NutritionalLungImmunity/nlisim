from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
from attr import attrib, attrs
from numpy import dtype

from nlisim.cell import CellData

datatype = Union[str, dtype, Type[Any]]


@attrs(kw_only=True)
class CellDataMaker:
    parent_class: CellData = attrib(default=CellData)
    fields: Dict[str, Tuple[datatype, int, Callable]] = attrib(factory=dict)

    def add_field(
            self,
            field_name: str,
            data_type: datatype,
            multiplicity: int = 1,
            initializer: Optional = None,
    ) -> 'CellDataMaker':
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

        class CreatedCellData(CellData):
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
