import enum
from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass
class Connection:
    mcs: jax.Array
    rssi: jax.Array
    selected: jax.Array
    tx_power: jax.Array


@jax.tree_util.register_pytree_node_class
class JaxEnum(enum.Enum):
    def __jax_array__(self):
        return jax.nn.one_hot(self.value, len(type(self)))

    def __init_subclass__(cls) -> None:
        jax.tree_util.register_pytree_node_class(cls)
        return super().__init_subclass__()

    def tree_flatten(self):
        children = (self.__jax_array__(),)
        aux_data = self
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return children[0]


class IsSelected(JaxEnum):
    FALSE = 0
    TRUE = 1
    NA = 2


class McsValue(JaxEnum):
    MCS_0 = 0
    MCS_1 = 1
    MCS_2 = 2
    MCS_3 = 3
    MCS_4 = 4
    MCS_5 = 5
    MCS_6 = 6
    MCS_7 = 7
    MCS_8 = 8
    MCS_9 = 9
    MCS_10 = 10
    MCS_11 = 11
    MCS_12 = 12
    MCS_13 = 13
    NA = 14


class TxPowerValue(JaxEnum):
    LVL_1 = 0
    LVL_2 = 1
    LVL_3 = 2
    LVL_4 = 3
    NA = 4
