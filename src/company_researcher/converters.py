
from company_researcher.models.models import ResearchResponse
from company_researcher.workflow.states import ResearchState


def research_state_to_response(state: ResearchState) -> ResearchResponse:
    """
    Convert ResearchState to ResearchResponse model.
    """
    return ResearchResponse(**state.model_dump())