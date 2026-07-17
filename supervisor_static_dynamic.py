from model.ResGCN_StaticDynamicSemanticChange import (
    ResGCN_StaticDynamicSemanticChange,
)
from supervisor import _EIN_BackboneOnly_supervisor


def EIN_ResGCN_StaticDynamicSemanticChange_supervisor(args):
    """Run the isolated static/dynamic semantic-change model."""

    return _EIN_BackboneOnly_supervisor(
        args,
        ResGCN_StaticDynamicSemanticChange,
        "ResGCN_StaticDynamicSemanticChange",
    )
