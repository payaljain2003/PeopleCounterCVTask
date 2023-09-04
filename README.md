# People Counter using Computer Vision

This repository contains code for a people counter using computer vision. 
It processes video data, identifies people in the frame, track them to maintain people count, 
also the entry and exit count from a particulare reference area.

## Getting Started

Follow these steps to set up and run the project:

### Clone the Repository

You can clone this repository using the following command:

```bash
git clone https://github.com/payaljain2003/PeopleCounterCVTask.git
```

Alternatively, you can download the repository as a ZIP file from the 'Code' dropdown menu and unzip it on your local machine.

### Prepare Data
1. Create an empty folder named 'Data' in the directory containing the main.py file.
2. Store the video file - 'MainGateLuminous.mp4' in the 'Data' folder.

### Download Required Files
Download the directories - detections and model_data containing multiple object tracking algorithm files and deepSORT network files from the provided Google Drive link and place them in the same directory as the main.py file:

[Link](https://drive.google.com/drive/folders/16XRNoUrT8zfIEteQiq2QgJ-q_RWKvkuG)


### Install Dependencies
Install the required Python dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Folder Structure
After completing the setup, your project directory should have the following structure:

![cropped](https://github.com/payaljain2003/PeopleCounterCVTask/assets/117659940/071d7a8c-ce97-48e2-9be6-3766c930555e)


### Usage
To execute the script and start counting people, run the following command:
```bash
python main.py
```



## Acknowledgment

I would like to acknowledge the following resources  that have been instrumental in the progress on this solution:

- [YouTube Channel 1](https://www.youtube.com/watch?v=jIRRuGN0j5E&t=341s)
- [YouTube Channel 2](https://www.youtube.com/watch?v=tryBL3xlk_c&t=1268s)   for their insightful tutorials.

- [ChatGPT](https://openai.com) 

- The Stack Overflow community for their contributions to problem-solving.


