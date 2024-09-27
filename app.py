from flask import Flask, render_template, request
from model import processor, model
from test_video import clip_baby
from matplotlib import pyplot as plt
from matplotlib import animation
from io import BytesIO
import base64

app = Flask(__name__)

# Категории, которые мы хотим, чтобы модель определила
categories = ["развлечения", "юмор", "спорт", "киберспорт", "образование", "путешествия", "детский контент", "прочие"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_title = request.form['video_title']
        video_description = request.form['video_description']

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

        # Generate the video preview
        fig = plt.figure()
        im = plt.imshow(clip_baby[0, :, :, :])
        plt.close()

        def init():
            im.set_data(clip_baby[0, :, :, :])

        def animate(i):
            im.set_data(clip_baby[i, :, :, :])
            return im

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=clip_baby.shape[0], interval=100)
        video_html = anim.to_html5_video()

        # Convert the video preview to base64 for display
        buffered = BytesIO()
        plt.savefig(buffered, format="png")
        video_preview = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return render_template('index.html', assistant_response=assistant_response, video_html=video_html, video_preview=video_preview)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)