from openai import OpenAI

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# For summary prompts
def build_summary_prompt(context):
    return f"""
    You are an assistant that summarizes the content.
    Use ONLY the provided context.
    Do not add new information.

    Context:
    {context}

    Task:
    Write a concise summary in 5-6 bullet points"""


# For QA Prompts
def build_qa_prompt(context, question):
    return f"""
    You are an assistant that answers question using ONLY the context.
    If the answer is not present, say "This topic is not in the provided notes."


    Context:
    {context}

    Question:
    {question}""" 

# For QUIZ prompt
def build_quiz_prompt(context):
    return f"""
    You are an assistant that create questions.
    Use ONLY provided context.

    Context:
    {context}

    Task:
    Create:
    - 3 multiple choice question
    - Each with 4 options
    - Mark the correct answer"""



def generate_response(mode, context_chunks, user_query=None):

    context_text = "\n\n".join(context_chunks)

    if mode == "summary":
        prompt = build_summary_prompt(context_text)

    elif mode == "qa":
        prompt = build_qa_prompt(context_text, user_query)

    elif mode == "quiz":
        prompt = build_quiz_prompt(context_text)

    else:
        raise ValueError("Invalid Mode")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system",
             "content": "You must strictly follow instructions."},

            {"role": "user",
             "content": prompt}
        ],
    )

    return response.choices[0].message.content