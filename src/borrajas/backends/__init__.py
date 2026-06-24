from .langgraph import react as react_langgraph
from .pydantic import react as pydantic_langgraph

VARIANTS = {
    'langgraph/react': react_langgraph,
    'pydantic/react': pydantic_langgraph
}