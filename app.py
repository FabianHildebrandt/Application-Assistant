# %% [markdown]
# ## Application assistant

# %% [markdown]
# ## Import

# %%
from docling.document_converter import DocumentConverter
import tqdm as notebook_tqdm
from pydantic import BaseModel, Field 
import os
from typing import Optional, Any, Literal, Dict, List, Tuple
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# from langfuse.callback import CallbackHandler
import gradio as gr
import contextlib
from io import StringIO
import docx
from pathlib import Path
import re
from typing import Union
from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# ## Telemetry/ Observability
# - Used for debugging, disabled on prod

# %%
langfuse_handler = None
# langfuse_handler = CallbackHandler()
if langfuse_handler:
    TRACING = True

# %% [markdown]
# ## API Key

# %%
USE_GOOGLE = False
try: 
    API_KEY = os.environ["NEBIUS_KEY"]
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-fast"
    ENDPOINT_URL = "https://api.studio.nebius.com/v1/"
    print("Using Nebius API")
except:
    try:
        API_KEY = os.environ["GOOGLE_API_KEY"]
        MODEL_NAME = os.environ["GOOGLE_DEPLOYMENT_NAME"]
        USE_GOOGLE = True
        print("Using Google API")
    except:
        raise ValueError("No NEBIUS API Key was found")

# %% [markdown]
# ## Structured outputs

# %%
class Feedback(BaseModel):
    feedback : str = Field("", description="Constructive feedback, stating the points that can be improved.") 
    quality_flag : Literal["PERFECT", "NEEDS IMPROVEMENT"] = Field("NEEDS IMPROVEMENT", description="Tells whether the cover letter needs to be reworked.")

class MultiStepPlan(BaseModel):
    reasoning : str = Field("", description="The multi-step reasoning required to break down the user query in a plan.")
    plan : List[Literal["critic_agent", "writer_agent", "recruiter_agent", "team_lead_agent","interview_agent"]] = Field("END", description="The list of agents required to fulfill the user request determined by the Orchestrator.")

# %% [markdown]
# ## Agent state

# %%
class AgentDescription(TypedDict):
    "Agent description containing the title, system prompt and description."
    title : str 
    description : str 
    system_prompt : str

class ApplicationAgentState(BaseModel):
    """State of the cover letter writer agent."""
    user_query : Optional[str] = Field("", description="User task for the agents to fulfill.")
    iterations : Optional[int] = Field(0, description="Counter for the evaluation-optimization loop of the cover letter.")
    max_iterations : Optional[int] = Field(3, description="Maximum number of iterations for the evaluation-optimization loop of the cover letter.")
    available_agents : Dict[str, AgentDescription] = Field(description="A dictionary of the available agents.")
    cv: Optional[str] = Field("",description="CV content parsed as a Markdown format from the document.")
    job_description : Optional[str] = Field("", description="Job description.")
    skills : Optional[str] = Field("", description="Required skills extracted from the job description")
    motivation : Optional[str] = Field("", description="You're desired job profiles and general motivation.")
    examples : Optional[str] = Field("", description="Examples of previous cover letters.")
    phase : Literal["PLAN", "EXECUTE", "ANSWER"] = Field("PLAN", description="Current phase of the agent")
    messages : List[Tuple[str,str]] = Field([], description="List of agent thoughts (agent, agent response).") 
    final_answer : str = Field("", description="Final answer generated after task execution.")
    plan : List[Literal["critic_agent", "writer_agent", "recruiter_agent", "team_lead_agent","interview_agent"]] = Field([],description="The current list of tasks to execute")
    cover_letter: Optional[str] = Field("", description="The cover letter for the specified job.")
    connected_skills : Optional[str] = Field("", description="Skills from the job description connected to previous working experience from the CV.")
    feedback : str = Field("", description="Written feedback from the critic agent regarding the cover letter.")

# %% [markdown]
# ## System prompts

# %%
general_prefix = """
You are part of a collaborative multi-agent system called the *Application Assistant*. 
This system consists of specialized agents working together to evaluate, improve, and support job applications. 
Each agent has a distinct role and expertise, and you are encouraged to consider and integrate relevant information or insights produced by other agents when available. 
Sharing context and building on each other's outputs will help generate higher-quality and more comprehensive results. 
Operate within your designated role, but feel free to utilize the shared context from other agents to inform your responses when appropriate.
"""


writer_prompt = """
You are a technical recruiter with years of experience helping candidates land roles in tech companies. Your job is to assist the user in crafting exceptional, personalized cover letters for tech positions.

Strict Style Guidelines:
1. Use professional but naturally flowing language appropriate for a proficient non-native English speaker.
2. Use the provided information by the user containing the CV, the job description, and if available previous motivation letters, the general motivation for the intended job change 
3. Stick to the provided information. Don't ever make up facts, experience or add any quantifiable numbers, that are not explicitly mentioned. 

Use the following structure to write a good motivation letter:
[STRUCTURE]
1. Opening paragraph
Goal: Grab attention, show enthusiasm, and state the role.
- Mention the specific job title. If possible, summarize long job title names and capture the most important aspects.
- Say why you’re excited about the company (specific project, mission, tech stack, etc.).
- Say how the job aligns with your personal goals

Example:
I’m thrilled to apply for the Backend Engineer role at Stripe. As someone who’s followed your API design philosophy and admired your commitment to developer experience, I’m excited about the opportunity to contribute to a team building scalable financial infrastructure.

2. Body Paragraph 1 – Your Value Proposition

Goal: Show how your experience matches their needs.
- Focus on 1–2 major accomplishments relevant to the role.
- Use number to underline your achievements, but only if they are explicitly mentioned in the povided information (e.g., “reduced latency by 30%”).
- Highlight how you have already used key technologies or skills they mentioned in the job ad.

Example:
At Plaid, I led a team optimizing real-time data sync across financial institutions, reducing sync errors by 40% and increasing transaction throughput by 25%. My experience with Go and distributed systems directly aligns with Stripe’s scale and architecture.

3. Body Paragraph 2 – Culture & Fit

Goal: Show alignment with company values and team dynamics.
- Briefly show why you’re a good cultural fit.
- Mention soft skills (e. g. collaboration, leadership, adaptability).
- Tie it back to something unique about the company.

Example:
Beyond the code, I thrive in collaborative, feedback-rich environments. I appreciate how Stripe emphasizes intellectual humility and long-term thinking — qualities that have defined my best work experiences.

4. Closing Paragraph – The Ask

Goal: Finish strong and express interest in the next step.
- Reiterate excitement.
- Thank them for considering you.
- Invite them to review your resume or portfolio.
- Mention you’re looking forward to the next step.

Example:
Thank you for considering my application. I’d love to bring my backend expertise and product-minded approach to Stripe. I’ve attached my resume and GitHub — I look forward to the opportunity to discuss how I can contribute to your team.
[END STRUCTURE]
Keep the cover letter short and limit it to one page.
"""

critic_prompt = """
You are a technical recruiter with years of experience helping tech companies identify strong candidates.

Your task is to review a candidate’s cover letter and provide brief, constructive feedback on its quality and fit. 

Use the following criteria to guide your evaluation:
- Does the writing flow naturally, as if written by a proficient non-native English speaker?
- Is the content clearly aligned with the job description and the candidate’s resume/motivation?
- Does the candidate effectively demonstrate the required skills and experience?
- Does the cover letter appear AI-generated? (e.g., overly polished language, unnatural structure, unusual punctuation like em-dashes)
- Is the cover letter a well-written continuous text instead of bullet points?

If any of the quality criteria is not fulfilled, the cover letter needs improvement.

Provide a short but constructive written feedback hightlighting the points of improvement.

Provide quality flag 
- PERFECT: if the cover letter matches most of the criteria. 
- NEEDS IMPROVEMENT: if improvements can be made
"""

recruiter_prompt = """
You are a professional recruiter agent responsible for critically and constructively evaluating how well a candidate fits a specific tech job role. 
Your assessment is based on the job description, the candidate’s CV, and their motivation letter (if available). 
Your goal is to identify the best possible match for the role and the company, ensuring a high-quality, data-driven, and fair evaluation process

Follow these steps:
1. Analyze the CV and Motivation Letter
Assess the following:
Relevant professional experience (projects, roles, industries)
Specific technical and soft skills
Educational background and certifications
Achievements and measurable impact
Motivation, intent, and career progression
Evidence of cultural and values alignment
Communication style and clarity

2. Identify Top 5 Required Skills from the Job Description
Extract the five most critical skills (technical or soft) based on job responsibilities and requirements.
Prioritize skills that are essential for success in the role and aligned with the company’s needs

3. Match Skills with Candidate Evidence
For each of the five skills:
Provide a concise bullet point explaining how the candidate demonstrates (or does not demonstrate) that skill.
Use specific excerpts or paraphrased references from the CV or motivation letter.
Indicate the strength of the match: Direct, Partial, or Missing.

Format:
•	[Skill 1]: Explanation with evidence
•	[Skill 2]: …

4. Provide a Score and Feedback (1–10)
Evaluate the candidate’s overall fit and give a score from 1 (poor fit) to 10 (excellent match).
Consider: experience relevance, skill alignment, motivation, cultural fit, and growth potential.

5. Summarize Key Findings (top 2-3 skills, biggest gaps and what to focus on during onboarding)

6. Recommendation: Move to interview / On the fence / Not a fit

Here is an example that shows you how to structure your thoughts:
[EXAMPLE]
Top 5 Required Skills from the Job Description
•	Project Management
•	Data Analysis
•	Stakeholder Communication
•	Strategic Thinking
•	Industry Knowledge (SaaS / B2B Tech)

Skill Match Analysis
•	Project Management: The candidate managed multiple cross-functional initiatives at Company A, including leading a product launch across three departments. Specifically noted: “Led a 6-month cross-functional project that launched a new product line, delivered on time and under budget.” Strong, direct fit.
•	Data Analysis: In their role at Company B, the candidate conducted monthly performance reporting and built dashboards using Excel and Power BI. CV excerpt: “Created automated reporting tools that reduced manual data processing by 40%.” Direct match, though more tools could enhance depth.
•	Stakeholder Communication: Candidate mentions managing client communication and reporting in a consulting context: “Presented findings and strategic options to C-suite clients in quarterly business reviews.” Shows experience with high-level communication, making this a strong fit.
•	Strategic Thinking: Candidate completed a strategic market entry plan as part of an MBA capstone and applied similar thinking in their work: “Developed 3-year strategic roadmap for internal operations streamlining, adopted by leadership.” Strong evidence of structured, long-term planning.
•	Industry Knowledge (SaaS / B2B Tech): The CV shows indirect exposure through consulting for tech firms, but no direct work in a SaaS product company. Slight mismatch: “Consulted B2B tech clients on pricing strategy”—relevant, but not hands-on experience.

Overall Score
8/10

Summary 
The candidate demonstrates a strong alignment with four of the five core skills, with especially solid experience in project management, communication, and strategic thinking. 
While their data analysis experience is good, it’s more operational than advanced. 
The main gap is a lack of direct SaaS or in-house product company experience, though consulting exposure softens this issue. 
Motivation letter expresses clear interest in joining a fast-growing B2B tech firm, citing alignment with company values and career growth in tech.
High potential to succeed with onboarding support in SaaS-specific environments.

Recommendation
Move to interview. 
[END EXAMPLE]

Guidelines:
Be objective, structured, and concise.
Use evidence-based reasoning.
Consider both technical proficiency and soft skills, as well as cultural and motivational alignment.
Focus on how the candidate’s profile aligns with the specific requirements and values of the company.
Your goal is to ensure a thorough, fair, and efficient evaluation that supports high-quality hiring
"""

team_lead_prompt = """
You are the team lead responsible for hiring a new member for your tech team. 
Your goal is to critically and constructively assess how well a candidate fits the specific job role, your team’s working style, and the broader company culture. 
Base your evaluation on the job description, the candidate’s CV, and their motivation letter (if available).
Your aim is to identify candidates who will excel technically, collaborate effectively, and contribute positively to your team’s success.


Follow these steps:
1.	Assess Practical Fit
•	Relevant hands-on experience
•	Technical or domain-specific skills
•	Ability to work independently and take ownership
•	Experience working in team environments (cross-functional, agile, remote, etc.)
•	Communication and collaboration style

2.	Identify Key Team-Relevant Skills from the Job Description
Extract the top 3–5 competencies essential for success in the team (e.g. problem-solving, initiative, tech stack experience, ownership, adaptability).

3.	Evaluate Evidence from CV and Motivation
For each selected skill
•	Describe how the candidate demonstrates it
•	Use specific examples or paraphrased references from the CV or motivation
•	Comment on the strength of the match (strong/partial/missing)
Format:
•	[Skill]: Explanation with evidence and strength of fit

4.	Provide a Team Fit Score (1–10)
Evaluate based on contribution potential, autonomy, collaboration, and mindset.
Give a short paragraph explaining the score.

5.	Summarize Fit for the Team
Briefly mention:
•	Strengths for immediate value
•	Potential gaps or onboarding needs
•	Whether the team would benefit from this person

6.	Recommendation
Move to interview / On the fence / Not a fit

Here is an example that should demonstrate how you can structure your thoughts:
[EXAMPLE]
Key Team-Relevant Skills from the Job Description
	•	English Language Proficiency
	•	Multicultural Team Experience
	•	Jira (Project Management Tool)
	•	Scrum or Agile Methodology
	•	Communication Skills

Skill Match Evaluation
	•	English Language Proficiency: CV and motivation letter are clearly written in fluent English. Candidate studied in an international Master’s program conducted entirely in English. Strong match.
	•	Multicultural Team Experience: No clear evidence of working in international or cross-cultural teams. All listed experience is local. Missing.
	•	Jira: Used Jira during a university group project to manage tasks and deadlines. Basic exposure, but partial match.
	•	Scrum or Agile Methodology: No mention of Scrum, Agile, or similar methodologies in either the CV or motivation. Missing.
	•	Communication Skills: Motivation letter is well-structured and reflects clear, professional language. Also gave presentations during university. Strong match for entry level.

Team Fit Score: 5.5 / 10

Summary for Team Fit
This candidate communicates clearly in English and has some basic familiarity with Jira. However, there’s a notable lack of hands-on experience in Agile/Scrum environments and no exposure to multicultural collaboration, which are key to this role. With strong onboarding and mentoring, the candidate might grow into the role, but would not be immediately effective in a globally distributed team setup.

Recommendation
Not a fit for this role — too many foundational gaps for a distributed, Scrum-based team.
[END EXAMPLE]

Guidelines:
Be objective, structured, and concise.
Use evidence-based reasoning.
Focus on both technical and interpersonal/teamwork skills.
Consider how the candidate’s profile aligns with your team’s current strengths, gaps, and working style.
Think about long-term growth and how the candidate could evolve within your team.
"""


interview_prompt = """
Please generate some nice interview questions based on the provided feedback. 

You are an AI interview question generator designed to help hiring teams prepare for structured, insightful, and role-relevant interviews. 
Your goal is to generate a diverse set of interview questions—technical, behavioral, and situational—tailored to the job description, the candidate’s CV, and (if available) their motivation letter. 
The questions should help assess both the candidate’s technical fit and their alignment with the team’s working style and company culture.

Instructions:
1. Analyze Inputs
- Please check first, if another agent has already provided feedback. In this case, skip the step 1 and proceed directly generating the interview questions.
- Carefully review the job description for key responsibilities, required skills, and team context.
- Examine the candidate’s CV and motivation letter for relevant experience, technical and soft skills, achievements, and evidence of cultural or value alignment.
- Identify Core Competencies
- Extract the top 5 critical skills or qualities needed for the role (these may be technical, interpersonal, or related to cultural fit).
2. Generate Interview Questions
- For each core skill or quality, create at least one targeted interview question.
Ensure a mix of question types:
- Technical questions (to assess knowledge and problem-solving)
- Behavioral questions (to explore past experiences and soft skills)
- Situational questions (to evaluate judgment and approach to real-world challenges)
- Where relevant, tailor questions to the candidate’s background (e.g., referencing specific projects or experience from their CV).
Ensure Depth and Relevance
- Questions should be open-ended, encourage detailed responses, and give insight into both expertise and working style.
- Include at least one question that probes motivation or cultural/team fit.
Output Format
- List questions grouped by skill/competency, clearly labeled.
- For each question, specify the type: [Technical], [Behavioral], or [Situational].
Examples:
Skill: Project Management
[Behavioral] Tell me about a time you managed a cross-functional project. What challenges did you face and how did you overcome them?
Skill: Team Collaboration
[Situational] If you joined a team with conflicting work styles, how would you approach building effective collaboration?
Skill: Technical Expertise (Python)
[Technical] Can you describe a complex Python project you’ve led? What were the key technical decisions and their impact?

Pool of general example questions: 
[Technical]
Describe how you would design a scalable system to handle millions of users. What architectural decisions would you make, and why?
Given a large dataset, how would you optimize a SQL query that is running slowly? What steps would you take to diagnose and resolve the issue?
Can you implement a function to detect cycles in a linked list? Explain your approach and its time/space complexity.
Walk me through how you would design a RESTful API for a ride-sharing application. What considerations would you prioritize for security and scalability?
Tell me about the most difficult bug you have fixed in the past six months. How did you identify and resolve it?
[Behavioral]
Tell me about a time you had to work with a difficult co-worker. How did you handle the interaction, and what was the outcome?
Describe a situation where you had to explain a complex technical problem to someone without a technical background. How did you ensure understanding?
Give an example of a time when you made a mistake because you failed to pay attention to detail. How did you recover from it?
What is the most challenging aspect of your current or most recent project? How did you tackle it?
Tell me about a time you had to influence someone to accept your point of view during a disagreement. What approach did you take, and what was the result?
[Situational]
Imagine you join a team where members have conflicting work styles. How would you approach building effective collaboration?
Suppose you are assigned a project with an extremely tight deadline and limited resources. What steps would you take to ensure successful delivery?
If you noticed a critical error in a project just before launch, what would you do? How would you communicate this to your team and stakeholders?
How would you handle a situation where a key team member suddenly leaves in the middle of a major project?
If you were given a task outside your current expertise, how would you approach learning and delivering results?

Guidelines:
Questions should be clear, concise, and tailored to the specific role and candidate profile.
Avoid generic or overly broad questions unless they directly serve the assessment goal or unless you don't have enough data available.
Focus on questions that will help the hiring team evaluate both immediate fit and long-term potential.
Your output should enable the hiring team to conduct a thorough, fair, and insightful interview that assesses the candidate’s overall suitability for the team and company.
"""

# %% [markdown]
# ### Available agents

# %%
recruiter_agent_description = AgentDescription(
    title="recruiter_agent",
    description="The recruiter agent provides general feedback about the applicant.",
    system_prompt=general_prefix+recruiter_prompt
)
writer_agent_description = AgentDescription(
    title="writer_agent",
    description="The writer agent can write a cover letter for the applicant.",
    system_prompt=general_prefix+writer_prompt
)
critic_agent_description = AgentDescription(
    title="critic_agent",
    description="The critic agent provides feedback for a written cover letter.",
    system_prompt=general_prefix+critic_prompt
)
interview_agent_description = AgentDescription(
    title="interview_agent",
    description="The interview question agent can generate interview questions for the candidate based on specialized feedback. It only makes sense to call the interview agent after feedback was provided to improve the results.",
    system_prompt=general_prefix+interview_prompt
)
team_lead_agent_description = AgentDescription(
    title="team_lead_agent",
    description="The team lead agent provides feedback about the applicant from the perspective of the team lead of the team of the open position.",
    system_prompt=general_prefix+team_lead_prompt
)
available_agents = {
    "recruiter_agent" : recruiter_agent_description,
    "writer_agent" : writer_agent_description,
    "critic_agent" : critic_agent_description,
    "interview_agent" : interview_agent_description,
    "team_lead_agent": team_lead_agent_description
}

# %% [markdown]
# ## Utilities

# %% [markdown]
# ### General utilities

# %%
def docling_extraction(source : str = "CV.pdf") -> str:
    "Extract CV and convert it to Markdown using docling"
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document.export_to_markdown()

def read_file_content(file: Union[str, Path]) -> str:
    file_path = Path(file)
    suffix = file_path.suffix.lower()

    if suffix == ".txt" or suffix == ".md":
        return file_path.read_text(encoding="utf-8")
    
    elif suffix == ".pdf":
        # Extract CV and convert it to Markdown using docling
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()

    elif suffix == ".docx":
        return "\n".join(p.text for p in docx.Document(file_path).paragraphs)

    else:
        return ""

def call_llm(system_prompt, user_prompt, response_format : Any = None) -> Any:
    """
    Call LLM with provided system prompt and user prompt and the response format that should be enforced
    """

    if USE_GOOGLE:
        llm = ChatGoogleGenerativeAI(
            model = MODEL_NAME,
            google_api_key = API_KEY,
            temperature = 0,
            max_tokens = None,
            timeout = None,
            max_retries = 2
        )
    else:
        llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=ENDPOINT_URL,
        max_completion_tokens=None,
        timeout=60,
        max_retries=0,
        temperature=0
        )
    
    if response_format is not None:  
        llm = llm.with_structured_output(response_format)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("user", "{user_prompt}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "system_prompt":system_prompt,
        "user_prompt": user_prompt
    })

    return response 

def serialize_messages(messages : List[Tuple[str,str]]) -> str:
    "Returns a formatted message history of previous messages"
    return "\n" +"\n".join(f"{role}:\n{content}" for role, content in messages)

def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

# %% [markdown]
# ### Gradio utilities

# %%
# Handle different result types cleanly
def type_conversion(obj : Any, type):
    "Return the object in a gradio compatible type"
    if isinstance(obj, type):
        result_dict = obj.model_dump()
    elif isinstance(obj, Dict):
        result_dict = obj
    else:
        # Handle possible dataclass or similar object
        try:
            result_dict = ApplicationAgentState.model_validate(obj).model_dump()
        except Exception as e:
            print(f"Error converting output of type {type(obj)}")

    return result_dict

# %% [markdown]
# ## Agents

# %% [markdown]
# ### Orchestrator agent
# 
# 

# %%
def orchestrator_agent(state: ApplicationAgentState) -> Command[Literal["recruiter_agent", "team_lead_agent", "writer_agent", "critic_agent", "interview_agent","final_answer_tool", END]]:
    """
    Central orchestration logic to determine which agent to call next based on the current state and results.
    """

    # check if the CV and a job description are provided by the user
    if not state.cv or not state.job_description:
        state.final_answer = "### ❗️ The application assisant needs more information about you and the desired position to provide high-quality results.\n" \
        "👈🏽 Please go to the tab 🤗 **Personal Information** and provide your CV and the job description."  
        return Command(
            goto=END,
            update=state
        )
    
    if state.phase == "PLAN":
        agent_descriptions = "\n".join([
            f"{agent.get("title")}\nDescription: {agent.get("description")}"
            for name, agent in state.available_agents.items()
        ])

        system_prompt = f"""You are an orchestrator agent, that delegates tasks to specialized agents based on a user query.

        Your rules:
        Given a user query, your task is to determine the next agents to call.
        You are only allowed to fulfil tasks that are related to application assistance (e. g. supporting to write cover letters, revise a CV, or generate a bunch of interview questions) AND match one of the following agent's capabilities.
        You are allowed to do multi-step plans, if required. 
        This is an overview of all available agents with their descriptions.

        Agents:
        {agent_descriptions}

        If a user query is not in the scope of the available agent capabilities, you can always call __end__ to end the workflow.
        
        Please provide short reasoning on why you choose an agent before doing your final choice.
    
        [EXAMPLES]
        USER QUERY: Please help me to generate a cover letter for this interesting position
        REASONING: The user is asking for support with a cover letter, so I might call the writer_agent specialized in writing cover letters.
        NEXT_AGENT: writer_agent

        USER_QUERY: I want to design an nice recepe to cook a meal for dinner
        REASONING: The user request doesn't match any of the agent capabilities and seems off-topic, so I should call __end__.
        NEXT_AGENT: __end__ 

        USER_QUERY: I want you to design me some realistic interview questions for my next interview with an HR recruiter and a team lead of Meta.
        REASONING: The user request asks for help with interview questions from the HR recruiter perspective. I might call the recruiter agent for that request first and then call the interview agent to generate some interview questions based on the feedback. 
        NEXT_AGENT: recruiter_agent, team_lead_agent, interview_agent

        [END EXAMPLES]
        """

        user_prompt = state.user_query
        state.messages.append(("user query", state.user_query))

        # call the orchestrator to select the next agent 
        response = call_llm(system_prompt, user_prompt, MultiStepPlan)
        print("="*40)
        print("🤖 ORCHESTRATOR PLAN")
        print("="*40)
        print(f"\n📝 Reasoning:\n{response.reasoning}\n")
        print("🔗 Planned Steps:")
        for i, step in enumerate(response.plan, 1):
            print(f"  {i}. {step}")
        print("="*40)
        print("⚙️ EXECUTE PLAN")
        print("="*40 + "\n")
        state.plan = response.plan
        state.phase = "EXECUTE"
    
    if len(state.plan) == 0 and state.phase == "EXECUTE":
        state.phase = "ANSWER"
        return Command(
            goto="final_answer_tool",
            update=state
        )
    if state.phase == "EXECUTE":
        # return next agent to call
        agent = state.plan.pop(0)
        try: 
            return Command(
                goto=agent,
                update=state)
        except Exception as e:
            print(f"Error executing the plan in step {i} - {step}: {e}")

    if state.phase == "ANSWER":
        state.phase = "PLAN"
        state.messages = [("orchestrator_agent", f"Final answer:\n{state.final_answer}")]
        
    # finally end the workflow execution
    return Command(goto=END, update=state)

# %% [markdown]
# ### Recruiter agent

# %%
def recruiter_agent(state : ApplicationAgentState) -> Command:
    "The recruiter agent has the task to provide feedback about the applicant from the perspective of a senior recruiter"

    agent_description = state.available_agents.get("recruiter_agent", {})
    system_prompt = agent_description.get("system_prompt", "You're a Senior recruiter agent that provides feedback about an applicant.")
    
    user_prompt = f"""
[JOB DESCRIPTION]
{state.job_description}
[END JOB DESCRIPTION]

[CV]
{state.cv}
[END CV]
    """

    if len(state.messages):
        user_prompt += "Other agents have already contributed to the task. Please use their contributions to improve your feedback for the applicant."
        user_prompt += serialize_messages(state.messages)

    user_prompt += "Provide feedback on the applicant from your perspective."

    print("The recruiter agent assesses your information to provide feedback 🧑🏼‍💻")
    response = call_llm(system_prompt, user_prompt).content
    print("The recruiter agent has provided feedback.")

    agent_contribution = ("recruiter_agent", response)
    state.messages.append(agent_contribution)
    
    return Command(
        goto="orchestrator_agent",
        update=state
    )

# %% [markdown]
# ### Team lead agent

# %%
def team_lead_agent(state : ApplicationAgentState) -> Command:
    "The team lead agent has the task to provide feedback about the applicant from the perspective of the team lead of the hiring team"

    agent_description = state.available_agents.get("team_lead_agent", {})
    system_prompt = agent_description.get("system_prompt", "You're a team lead and you want to hire a new person. Provide feedback on the applicant.")
    
    user_prompt = f"""
[JOB DESCRIPTION]
{state.job_description}
[END JOB DESCRIPTION]

[CV]
{state.cv}
[END CV]
    """

    if len(state.messages):
        user_prompt += "Other agents have already contributed to the task. Please use their contributions to improve your feedback for the applicant."
        user_prompt += serialize_messages(state.messages)

    user_prompt += "Provide feedback on the applicant from your perspective."

    print("The team lead agent assesses your information to provide feedback 🧑🏼‍💻")
    response = call_llm(system_prompt, user_prompt).content
    print("The team lead agent has provided feedback.")

    agent_contribution = ("team_lead_agent", response)
    state.messages.append(agent_contribution)
    
    return Command(
        goto="orchestrator_agent",
        update=state
    )

# %% [markdown]
# ### Writer agent

# %%
def writer_agent(state: ApplicationAgentState) -> Command[Literal["critic_agent"]]:
    """Generate a cover letter using job description, CV, motivation, and skill match."""
    
    agent_description = state.available_agents.get("writer_agent", {})
    system_prompt = agent_description.get("system_prompt", "You're a critic agent that provides helpful feedback for cover letters.")
    
    user_prompt = f"""
[JOB DESCRIPTION]
{state.job_description}
[END JOB DESCRIPTION]

[CV]
{state.cv}
[END CV]
    """

    if state.motivation != "":
        user_prompt += f"\nThis is my general motivation and my desired job profiles:\n[MOTIVATION]{state.motivation}\n[END MOTIVATION]\n"

    if state.examples != "":
        user_prompt += f"Use the following examples of previous cover letters to adapt to my personal writing style:\n[EXAMPLES]\n{state.examples}\n[END EXAMPLES]\n"

    if len(state.messages):
        user_prompt += "Other agents have already contributed to the task. Please use their contributions to improve your writing."
        user_prompt += serialize_messages(state.messages)
    
    user_prompt += "Write a professional cover letter in under 300 words in the language of the job description."
    print("The writer agent writes the cover letter ✏️")
    agent_contribution = ("writer_agent", call_llm(system_prompt, user_prompt).content)
    print("The writer agent has completed a draft for your cover letter 📝")
    state.messages.append(agent_contribution)

    return Command(
        goto="critic_agent",
        update=state
    )



# %% [markdown]
# ### Interview agent

# %%
def interview_agent(state : ApplicationAgentState) -> Command[Literal["orchestrator_agent"]]:
    "Agent to generate interview questions based on the provided feedback of a recruiter of team lead agent."

    agent_description = state.available_agents.get("interview_agent", {})
    system_prompt = agent_description.get("system_prompt", "You're an interview agent, that generates questions for an applicant in a job interview.")
    
    user_prompt = f"""
[JOB DE SCRIPTION]
{state.job_description}
[END JOB DESCRIPTION]

[CV]
{state.cv}
[END CV]
    """

    if len(state.messages):
        user_prompt += "Other agents have already contributed to the task. Please use their contributions to improve the quality of your interview questions."
        user_prompt += serialize_messages(state.messages)

    print("The interview agent is generating a set of high-quality interview questions ❓")
    response =  call_llm(system_prompt, user_prompt).content
    print("The interview agent has generated a set of interview questions. ")

    agent_contribution = ("interview_agent", response)
    state.messages.append(agent_contribution)

    return Command(
        goto="orchestrator_agent",
        update=state
    )


# %% [markdown]
# ### Critic agent

# %%
def critic_agent(state: ApplicationAgentState) -> Command[Literal["writer_agent","orchestrator_agent"]]:
    "Provide feedback for a previously written cover letter."

    agent_contribution = ("critic_agent", "The critic agent was here..")

    user_prompt = f"""
[JOB DESCRIPTION]
{state.job_description}
[END JOB DESCRIPTION]

[CV]
{state.cv}
[END CV]
You are only allowed to make suggestions like quantifying experience if the required information was provided in the CV and is based on the actual experience. 
"""
    if state.motivation != "":
        user_prompt += f"\nThis is my general motivation and my desired job profiles:\n[MOTIVATION]{state.motivation}\n[END MOTIVATION]\n"

    if state.examples != "":
        user_prompt += f"Use the following examples of previous cover letters to adapt to my personal writing style:\n[EXAMPLES]\n{state.examples}\n[END EXAMPLES]\n"

    if len(state.messages):
        user_prompt += "Other agents have already contributed to the task. Please use their contributions to provide feedback to the most recent cover letter."
        user_prompt += serialize_messages(state.messages)

    agent_description = state.available_agents.get("critic_agent", {})
    system_prompt = agent_description.get("system_prompt", "You're a critic agent that provides helpful feedback for cover letters.")
    
    print("The critic agent revises the cover letter 🔎📝")
    response = call_llm(system_prompt, user_prompt, Feedback)
    print("The critic agent revised the cover letter ✅")

    state.iterations += 1

    if response.quality_flag == "PERFECT" or state.iterations == state.max_iterations:
        next_step = "orchestrator_agent"
    else:
        next_step = "writer_agent"
    
    print(f"[{state.iterations}. Iteration]\nFEEDBACK: {response.quality_flag}")

    agent_contribution = ("critic_agent", f"{response.feedback}")
    
    state.messages.append(agent_contribution)

    return Command(
        goto=next_step,
        update=state
    )

# %%
def final_answer_tool(state : ApplicationAgentState) -> Command[Literal["orchestrator_agent"]]:
    "Final answer tool is invoked to formulate a final answer based on the agent message history"

    system_prompt = f"""
    You're a helpful application assistant and your role is to provide a concise final answer will all the relevant details to answer the user query.

    Your final answer WILL HAVE to contain these parts:
    ### Task outcome
    ## Final answer
    """

    formatted_history = serialize_messages(state.messages)
    
    user_prompt = f"""
    ---
    Task:
    {state.user_query}
    ---
    Agent call history:
    {formatted_history}
    """

    final_answer = call_llm(system_prompt, user_prompt).content

    if isinstance(final_answer, str):
        final_answer = strip_think_blocks(final_answer)
        state.final_answer = final_answer
    
    return Command(
        goto="orchestrator_agent",
        update=state
    )


# %% [markdown]
# ## Build Graph

# %%
builder = StateGraph(ApplicationAgentState)

builder.add_node(orchestrator_agent)
builder.add_node(recruiter_agent)
builder.add_node(team_lead_agent)
builder.add_node(writer_agent)
builder.add_node(critic_agent)
builder.add_node(interview_agent)
builder.add_node(final_answer_tool)

builder.add_edge(START, "orchestrator_agent")

graph = builder.compile()

# %% [markdown]
# ## Gradio functions

# %% [markdown]
# ### Gradio function - Extract information

# %%
def extract_information(
    state_dict,
    cv_file,
    job_description_file,
    motivation_file,
    examples_file,
    max_iterations: int
) -> tuple[str, Dict, bool]:
    """
    Run the extraction pipeline and return output logs + state as a dict.
    """
    output_text = ""

    try: 
        cv_content = read_file_content(cv_file)
        job_description_content = read_file_content(job_description_file)
        motivation_content = read_file_content(motivation_file) if motivation_file else ""
        examples_content = read_file_content(examples_file) if examples_file else ""
        output_text += "Successfully extracted input."      
    except Exception as e:
        output_text += f"Reading input files failed: {str(e)}"
        return output_text, None, False
    
    state = ApplicationAgentState.model_validate(state_dict)
    state.cv = cv_content
    state.job_description = job_description_content
    state.motivation = motivation_content
    state.examples = examples_content 
    
    state_dict = type_conversion(state, ApplicationAgentState)
    
    return output_text, state_dict, True

# %%
def call_orchestrator(state_dict : Dict, user_query : str):
    "Function prototype to call the orchestrator agent"
    state = ApplicationAgentState.model_validate(state_dict)

    state.user_query = user_query
    # Capture print outputs
    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        # if TRACING:
        #     result = graph.invoke(input=state, config={"callbacks": [langfuse_handler]})
        # else:
           # result = graph.invoke(input=state)
        result = graph.invoke(input=state)

    result_dict = type_conversion(result, ApplicationAgentState)

    output_text = buffer.getvalue()

    return output_text, result_dict, True

# %% [markdown]
# ## Gradio Interface

# %%
with gr.Blocks() as application_agent_server:
    gr.Markdown("# 🏆 Application Assistant")

    with gr.Row():
        with gr.Column(scale=1):  # Image on the far left
            gr.Image(value="application_assistant.png", container=False, show_download_button=False, show_fullscreen_button=False)

        with gr.Column(scale=4):  # Markdown starts next to the image
            gr.Markdown("## Your happy AI assistant 🦛🤗 helps you to land your next dream job.")
            gr.Markdown("Just provide some information about yourself and the oppurtunity and ask the assistant for help.")

    state_dict = gr.State(value=ApplicationAgentState(available_agents=available_agents).model_dump())
    extraction_successful = gr.State(value=False)
    
    with gr.Tabs():
        with gr.TabItem("🤗 Personal information"):
            gr.Markdown("### 🍉 Feed the application agent with some information about yourself and the opportunity.")

            with gr.Row():
                cv_file = gr.File(value = "CV.md", label="Upload CV", file_types=[".pdf", ".txt", ".docx", ".md"], height=150)
                job_description_file = gr.File(value= "job-description.txt", label="Upload Job Description", file_types=[".pdf", ".txt", ".docx", ".md"], height=150)

            gr.Markdown("### Optional information to improve the quality of the results.")
            with gr.Row():
                motivation_file = gr.File(value= "motivation.txt", label="Upload General Motivation", file_types=[".pdf", ".txt", ".docx",".md"], height=150)
                examples_file = gr.File(value= "examples.txt", label="Upload previous Cover Letters", file_types=[".pdf", ".txt", ".docx",".md"], height=150)
            
            with gr.Accordion("Advanced options", open=False):
                max_iterations = gr.Number(label="Number of refinement iterations", value=2, precision=0)


            extract_button = gr.Button("Extract your information", variant="primary")

            extract_console_output = gr.Textbox(label="Logs / Console Output")

            @gr.render(inputs=[extraction_successful, state_dict])
            def show_contents(extraction_flag, state_dict):
                if extraction_flag:
                    cv = state_dict.get("cv","")
                    job_description = state_dict.get("job_description","")
                    motivation = state_dict.get("motivation","")
                    examples = state_dict.get("examples", "")
                    
                    with gr.Accordion("CV", open=False):
                        gr.Markdown(cv)
                    with gr.Accordion("Job Description", open=False):
                        gr.Markdown(job_description)
                    if motivation:
                        with gr.Accordion("Motivation", open=False):
                            gr.Markdown(motivation)
                    if examples:
                        with gr.Accordion("Previous cover letters", open=False):
                            gr.Markdown(examples)
            

        extract_button.click(
            fn=extract_information,
            inputs=[state_dict, cv_file, job_description_file, motivation_file, examples_file, max_iterations],
            outputs=[extract_console_output, state_dict, extraction_successful]
        )

        with gr.TabItem("🤖 Q&A Chatbot"):
            examples = """ℹ️ **Examplary queries**
- Write a cover letter for me.
- I have an interview with the team lead of the hiring team. Please provide some tailored interview questions.
- Please provide some feedback from the perspective of a recruiter on my CV and previous experience.
            """
            gr.Markdown(examples)
            user_query = gr.Textbox(label="Ask your question here", value="Generate a cover letter", interactive=True)
            button = gr.Button("Ask the application assistant 🦛🤗", variant="primary")
            qa_orchestrator_completed = gr.State(value=False)


            @gr.render(inputs=[qa_orchestrator_completed,state_dict])
            def show_qa_results(qa_flag, state_dict):
                if qa_flag: 
                    gr.Markdown(state_dict.get("final_answer", "❗️ No final answer provided: please check the logs."))
                    # reset the flag after orchestor was called
                    # qa_orchestrator_completed.value = False 
                else:
                    gr.Markdown("### Call the application assistant to get some results.")
            output_logs = gr.Textbox(label="Logs/ Console Output")

            # def chat_fn(message : str, history : List, state_dict : Dict):
            #     # Call your orchestrator logic
            #     output_logs, new_state_dict, success = call_orchestrator(state_dict, message)
            #     # Extract the assistant's reply from state_dict or output_logs
            #     reply = new_state_dict.get("final_answer", "Sorry, I couldn't process your request.")
            #     history = history + [(message, reply)]
            #     return reply, history, new_state_dict

            # chat = gr.ChatInterface(
            #     fn=chat_fn,
            #     additional_inputs=[state_dict],  # Pass state_dict as input
            #     additional_outputs=[state_dict], # Return updated state_dict
            #     # examples=[
            #     #     "Write a cover letter for me.",
            #     #     "I have an interview with the team lead of the hiring team. Please provide some tailored interview questions.",
            #     #     "Please provide some feedback from the perspective of a recruiter on my CV and previous experience."
            #     # ],
            #     title="Ask the application assistant"
            # )
            def reset_elements(qa_flag : bool, output_logs : str) -> bool:
                return False, "Generating response"

            button.click(
                fn=reset_elements,
                inputs=[qa_orchestrator_completed, output_logs],
                outputs=[qa_orchestrator_completed, output_logs]
            ).then(
                fn=call_orchestrator,
                inputs=[state_dict, user_query],
                outputs=[output_logs, state_dict, qa_orchestrator_completed]
            )

        with gr.TabItem("🔎 What's under the hood?"):
            gr.Markdown("## Details")
            with gr.Row():
                with gr.Column(scale=1):  # Image on the far left
                    gr.Image(value="mas_architecture.png", container=False, label="Architecture")

                with gr.Column(scale=2):  # Markdown starts next to the image
                    gr.Markdown("There is a LangGraph-powered multi-agent system under the hood using an orchestrator approach to plan and route the requests.")
                    gr.Markdown("Each agent is specialized performing application-related tasks.")

if __name__ == "__main__":
    application_agent_server.launch(mcp_server=True)




