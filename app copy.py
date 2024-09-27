from model import processor, model
from test_video import clip_baby


# Категории, которые мы хотим, чтобы модель определила
categories = [ "развлечения", "юмор", "спорт", "киберспорт", "образование", "путешествия", "детский контент", "прочие"]

# Get the video title and description from user input
video_title = input("Введите название видео: ")
video_description = input("Введите описание видео:  ")

# Prepare the conversation data
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Определите, к какой из следующих категорий относится это видео: {', '.join(categories)}. Ответьте одним или несколькими словами, если видео не относится ни к одной из категорий, укажите предполагаемую категорию."},
            {"type": "video", "title": video_title, "description": video_description},
        ],
    },
]

# Prepare the input data for the model
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor([prompt], videos=[clip_baby], padding=True, return_tensors="pt").to(model.device)
# Call the model to generate the top 10 categories
generate_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_p": 0.5}
output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True)

# Extract the relevant part of the output
assistant_response = generated_text[0].split("ASSISTANT: ")[1]

print(f"Модель определила, что это видео относится к категории: {assistant_response}")