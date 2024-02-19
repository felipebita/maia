from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from openai import OpenAI as OPA
from langchain.prompts import PromptTemplate

def prompt_enhancer(model,temperature,text):
    """
    Enhances user prompts for generative AI image creation models.

    Parameters:
    - model (str): Name or identifier of the language model.
    - temperature (float): Temperature parameter for language model generation.
    - text (str): User prompt to be enhanced.

    Returns:
    - enhanced_prompt (str): Enhanced prompt for the model.

    Example:
    >>> model = "gpt-3.5-turbo"
    >>> temperature = 0.7
    >>> text = "A beautiful landscape with mountains and a sunset."
    >>> enhanced_prompt = prompt_enhancer(model, temperature, text)
    >>> print(enhanced_prompt)
    'You are a professional assistant that helps people create images in a generative AI model. Sometimes their prompts lack information. Enhance the prompt to be simpler, not exceeding 400 words. User prompt: A beautiful landscape with mountains and a sunset. Your prompt:'
    """
    llm = ChatOpenAI(model_name=model,temperature=temperature)
    prompt = PromptTemplate(input_variables=["image_desc"], template="You are a professional assistant that help people who wants to create images in a generative AI model. Some times their prompt is not good and lack information. This way you are going to receive their promtp and enhance it to a better prompt for the model. Make it simples, not more than 400 words. User prompt: {image_desc}. Your prompt:")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text)

def get_images_2(model,prompt,size,n=1):
    """
    Generates images using a generative AI model.

    Parameters:
    - model (str): Name or identifier of the image generation model.
    - prompt (str): Prompt for image generation.
    - size (str): Size of the generated image (e.g., "medium", "large").
    - n (int): Number of images to generate. Default is 1.

    Returns:
    - caption (list): List of captions for the generated images.
    - urls (list): List of URLs for the generated images.

    Example:
    >>> model = "my_image_model"
    >>> prompt = "A beautiful landscape with mountains and a sunset."
    >>> size = "medium"
    >>> n = 3
    >>> captions, urls = get_images_2(model, prompt, size, n)
    >>> print(captions)
    [1, 2, 3]
    >>> print(urls)
    ['https://image1_url']
    """
    client = OPA()
    response = client.images.generate(model=model,prompt=prompt,size=size,n=n)
    caption =[]
    urls = []
    if n > 1:
        for i in range(0,n-1):
            caption =[caption,i+1]
            urls = [urls,response.data[i].url]          
            return caption, urls
    else:
        return response.data[0].url

def get_images_3(model,prompt,size,quality,style):
    """
    Generates images using a generative AI model with specified quality and style.

    Parameters:
    - model (str): Name or identifier of the image generation model.
    - prompt (str): Prompt for image generation.
    - size (str): Size of the generated image (e.g., "medium", "large").
    - quality (str): Quality of the generated image (e.g., "high", "medium", "low").
    - style (str): Style of the generated image.

    Returns:
    - url (str): URL for the generated image.

    Example:
    >>> model = "my_image_model"
    >>> prompt = "A beautiful landscape with mountains and a sunset."
    >>> size = "medium"
    >>> quality = "high"
    >>> style = "abstract"
    >>> image_url = get_images_3(model, prompt, size, quality, style)
    >>> print(image_url)
    'https://generated_image_url'
    """
    client = OPA()
    response = client.images.generate(model=model,prompt=prompt,size=size,quality=quality,style=style)
    return response.data[0].url 