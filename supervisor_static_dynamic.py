from model.BiGCN_StaticDynamicSemanticChange import (
    BiGCN_StaticDynamicSemanticChange,
)
from model.ResGCN_StaticDynamicSemanticChange import (
    ResGCN_StaticDynamicSemanticChange,
)
from supervisor import _EIN_BackboneOnly_supervisor


def EIN_BiGCN_StaticDynamicSemanticChange_supervisor(args):
    """Run the isolated BiGCN static/dynamic semantic-change model."""

    return _EIN_BackboneOnly_supervisor(
        args,
        BiGCN_StaticDynamicSemanticChange,
        "BiGCN_StaticDynamicSemanticChange",
    )


def EIN_ResGCN_StaticDynamicSemanticChange_supervisor(args):
    """Run the isolated static/dynamic semantic-change model."""

    return _EIN_BackboneOnly_supervisor(
        args,
        ResGCN_StaticDynamicSemanticChange,
        "ResGCN_StaticDynamicSemanticChange",
    )
