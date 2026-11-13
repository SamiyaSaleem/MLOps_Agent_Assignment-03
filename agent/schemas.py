from typing import Any
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
	question: str = Field( ..., min_length=1, max_length=2000,)
	
class ToolEvidence(BaseModel):
	tool:str
	status:str
	
class AgentResponse(BaseModel):
	status:str
	answer:str
	tools_used: list[str] = Field(default_factory=list)
	evidence: list[ToolEvidence] = Field(default_factory=list)
	error: str | None = None
	
class ToolExecutionResult(BaseModel):
	status:str
	tool:str
	result:dict[str, Any]
