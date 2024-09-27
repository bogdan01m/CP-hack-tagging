from flask import Flask, render_template, request, redirect, url_for
from model import processor, model
from read_and_play import read_video_pyav
from matplotlib import pyplot as plt
from matplotlib import animation
from io import BytesIO
import base64
import pandas as pd
import av
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Загружаем данные из IAB_tags.csv
iab_tags = pd.read_csv('IAB_tags.csv')

# Формируем список категорий с учетом иерархической структуры
categories = []
for i, row in iab_tags.iterrows():
    # Уровень 1 (родительская категория)
    if not pd.isna(row['Уровень 1 (iab)']):
        categories.append(row['Уровень 1 (iab)'])
    # Уровень 2 (подкатегория Уровня 1)
    if not pd.isna(row['Уровень 2 (iab)']):
        categories.append('{} > {}'.format(row['Уровень 1 (iab)'], row['Уровень 2 (iab)']))
    # Уровень 3 (подкатегория Уровня 2)
    if not pd.isna(row['Уровень 3 (iab)']):
        categories.append('{} > {} > {}'.format(row['Уровень 1 (iab)'], row['Уровень 2 (iab)'], row['Уровень 3 (iab)']))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_title = request.form['video_title']
        video_description = request.form['video_description']
        video_file = request.files['video_file']

        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)

            # Чтение видео с помощью PyAV
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip_baby = read_video_pyav(container, indices)

            # Преобразуем категории в строку для промпта
            categories_str = ', '.join(categories)

            # Prepare the conversation data
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Определите, к какой из следующих категорий относится это видео: {categories_str}. Ответьте одним или несколькими словами. Если видео не относится ни к одной из категорий, укажите предполагаемую категорию. Категории и их подгруппы должны браться только из предоставленного списка."},
                        {"type": "video", "title": video_title, "description": video_description},
                    ],
                },
            ]

            # Примеры для модели
            examples = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Определите, к какой из следующих категорий относится это видео: Транспорт, Спорт, Технологии. Ответьте одним или несколькими словами. Если видео не относится ни к одной из категорий, укажите предполагаемую категорию. Категории и их подгруппы должны браться только из предоставленного списка."},
                        {"type": "video", "title": "Обзор нового автомобиля", "description": "Обзор нового автомобиля от компании XYZ."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Транспорт > Типы кузова автомобиля > Седан"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Определите, к какой из следующих категорий относится это видео: Транспорт, Спорт, Технологии. Ответьте одним или несколькими словами. Если видео не относится ни к одной из категорий, укажите предполагаемую категорию. Категории и их подгруппы должны браться только из предоставленного списка."},
                        {"type": "video", "title": "Как играть в футбол", "description": "Уроки по игре в футбол для начинающих."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Спорт > Футбол"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Определите, к какой из следующих категорий относится это видео: Транспорт, Спорт, Технологии. Ответьте одним или несколькими словами. Если видео не относится ни к одной из категорий, укажите предполагаемую категорию. Категории и их подгруппы должны браться только из предоставленного списка."},
                        {"type": "video", "title": "Поверь мне, подруга I #10 I Как стать популярной в школе", "description": "Поговорим о том, как вести себя в новой школе."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Хобби и интересы"},
                    ],
                },
            ]

            # Объединение примеров и текущего запроса
            conversation.extend(examples)

            # Prepare the input data for the model
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor([prompt], videos=[clip_baby], padding=True, return_tensors="pt").to(model.device)

            # Call the model to generate categories
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
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
