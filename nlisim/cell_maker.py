import typing
from typing import Any, Callable, Dict, List, Tuple, Type

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellField, CellFields
from nlisim.util import Datatype, name_validator, upper_first


@attrs(order=False)
class GeneratedCellData(CellData):
    fields: List[Tuple[CellField, Callable]] = attrib()
    dtype: np.dtype = attrib()

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + tuple(
            [
                kwargs[field_name] if field_name in kwargs else initializer()
                for (field_name, *_), initializer in cls.fields
            ]
        )


@attrs(kw_only=True)
class CellDataFactory:
    name: str = attrib(validator=name_validator)
    parent_class: Type[CellData] = attrib(default=CellData)
    fields: Dict[str, Tuple[Datatype, int, Callable]] = attrib(factory=dict)

    def add_field(
        self,
        field_name: str,
        data_type: Datatype,
        initializer: Any,
        multiplicity: int = 1,
    ) -> 'CellDataFactory':
        if field_name in self.fields:
            raise RuntimeError(
                f"Module {self.name}: Trying to create the field {field_name} twice!"
            )
        if multiplicity < 1:
            raise RuntimeError(
                f"Module {self.name}: "
                f"Cannot create a field {field_name} of multiplicity less than 1."
            )

        if initializer is None:
            initializer_fn = default_initializer(data_type, field_name, multiplicity)
        elif callable(initializer):
            initializer_fn = initializer
        else:
            # make it callable
            def initializer_fn():
                return initializer

        self.fields[field_name] = (data_type, multiplicity, initializer_fn)

        return self

    def build(self) -> Type[CellData]:
        new_fields: CellFields = [
            (field_name, data_type, multiplicity) if multiplicity > 1 else (field_name, data_type)
            for field_name, (data_type, multiplicity, initializer) in self.fields.items()
        ]
        fields = self.parent_class.FIELDS + new_fields

        class_name = upper_first(self.name) + 'CellData'

        new_class: Type[CellData] = typing.cast(
            Type[CellData],
            attr.make_class(
                name=class_name,
                attrs={
                    'fields': attrib(default=fields, type=List[Tuple[CellField, Callable]]),
                    'dtype': attrib(factory=lambda: np.dtype(fields, align=True), type=np.dtype),
                },
                bases=(CellData,),
                kw_only=True,
                repr=False,
                order=False,
            ),
        )
        return new_class


def default_initializer(data_type: Datatype, field_name: str, multiplicity: int):
    # use a default
    if np.issubdtype(data_type, np.number):
        if multiplicity > 1:
            return lambda: np.zeros(shape=multiplicity, dtype=data_type)
        else:
            if np.issubdtype(data_type, np.inexact):
                return lambda: 0.0
            elif np.issubdtype(data_type, np.integer):
                return lambda: 0
    elif np.issubdtype(data_type, np.bool_):
        if multiplicity > 1:
            return lambda: np.zeros(shape=multiplicity, dtype=bool)
        else:
            return lambda: False
    else:
        raise RuntimeError(
            f"For the field {field_name} of type {data_type} with multiplicity {multiplicity}, "
            "you must provide an explicit initializer"
        )
