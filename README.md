🖐️ Gesture Language Translator
📖 Overview

Gesture Language Translator is an AI-based project that converts sign language gestures into text and also performs the reverse translation — generating sign language images from input text.
This project bridges the communication gap between hearing-impaired individuals and others by using computer vision and deep learning to recognize hand gestures in real-time.

🚀 Features

🔤 Real-time sign language to text conversion using webcam input.

🖼️ Text to sign image generation — enter any word (like “hello”) to see its sign language image.

⚙️ Built using Python, OpenCV, TensorFlow, and Flask.

💻 Simple and interactive web-based interface.

🧰 Tech Stack

Programming Language: Python

Libraries & Tools: OpenCV, TensorFlow, Flask, NumPy, Pandas

Frontend: HTML, CSS

Environment Management: Conda

⚙️ Installation & Setup

Follow these steps to set up and run the project locally 👇

1️⃣ Clone the repository
git clone https://github.com/your-username/Gesture-Language-Translator.git
cd Gesture-Language-Translator

2️⃣ Initialize Conda environment

Depending on your terminal, initialize Conda:

conda init powershell
# OR
conda init bash

3️⃣ Create and activate environment
conda activate gesture

4️⃣ Install dependencies
pip install -r requirements.txt

5️⃣ Verify Conda environments
conda info --envs


Make sure the gesture environment is active.

6️⃣ Run the application
python app.py


Then open your browser and navigate to the local server URL shown in the terminal (usually http://127.0.0.1:5000/).

🎯 How It Works

The webcam captures hand gestures.

The model processes frames using OpenCV and classifies gestures using a CNN model trained with TensorFlow.

The recognized gesture is displayed as text output.

Users can also type text and get corresponding sign language images.

🧑‍💻 Project Lead

Likhitha Penumudi
📍 Guntur, Andhra Pradesh
📧 likhithapenumudi@gmail.com

🔗 LinkedIn
 | GitHub

💡 Future Enhancements

Add voice output for translated text.

Support for sentence-level gesture translation.

Integrate text-to-video sign language generation.

🏆 Acknowledgment

Developed as part of an academic project showcasing AI and Deep Learning applications for accessibility and inclusivity.
