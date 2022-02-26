# asl-filter
This is a data science project that applies an ASL translation filter over a virtual camera which can then be used in teleconferences such as zoom, discord and skype. It tracks landmarks on your body such as your face, hands and pose, feedingx the data into a machine learning model to make a prediction on what word is being signed.

## Quick start guide

### Prerequisites
Download [OBS Studio](https://obsproject.com/download)

### Usage
1. Download repo
```
git clone https://github.com/jerremyng/asl-filter.git
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run python script
```
python deploy_filter.py
```

4. Select virtual camera from your teleconference app (zoom,skype,discord)
