# 2-Minute Demo Video Explanation Script
## HackZen 2.0 – Day 3 Computer Vision Track
### Project: AI-Based Real-Time Face Detection and Counting System
**Presenter Role**: Second-Year B.Tech IT Student  
**Target Duration**: 2 Minutes (120 Seconds)  
**Format**: Screen Recording with Voiceover + Webcam Feed

---

### [0:00 - 0:20] 1. Introduction & Hook
**Visual on Screen**: Title Slide or Terminal with project folder open, showing `main.py`.  
**Speaker**:  
> "Hello judges and mentors of HackZen 2.0!  
> I am a second-year B.Tech Information Technology student, and today I am excited to present my project for the Day 3 Computer Vision Track: the **AI-Based Real-Time Face Detection and Counting System**.  
> In public venues, classrooms, and automated entry points, tracking human footfall accurately without expensive GPU hardware or privacy risks is a major challenge. Let’s see how this application solves that problem."

---

### [0:20 - 0:45] 2. Live Demonstration
**Visual on Screen**: Run command `python main.py` in terminal. The camera window opens smoothly. You appear in front of the camera with a green bounding box around your face, a tag saying `Face #1`, and the bottom bar reading `Total Faces Detected: 1`.  
**Speaker**:  
> "As you can see, I simply execute `python main.py`. Instantly, the application accesses my laptop webcam and launches this clean, high-contrast interface.  
> Right away, the system detects my face, encloses it in a bright green bounding box with tech-style corner accents, and assigns it an indexed label: `Face #1`.  
> At the bottom, our heads-up display dynamically calculates and displays: `Total Faces Detected: 1`, along with our live framerate cruising smoothly at over 30 FPS."

---

### [0:45 - 1:15] 3. Multi-Person Detection & Robustness Test
**Visual on Screen**: A friend/classmate steps into the camera frame or you hold up a picture/phone photo with a face on it. The screen instantly draws a second green box `Face #2`, and the counter updates to `Total Faces Detected: 2`. When they step away, it immediately drops back to `1`.  
**Speaker**:  
> "Notice what happens when another person enters the frame—or if I introduce an additional face. The system detects multiple faces simultaneously, tags them individually, and instantly updates the live counter to `Total Faces Detected: 2`.  
> When the person steps out, the counter immediately resets in real time with zero lag.  
> The system operates entirely on the CPU with under 80 MB of RAM, without sending any video data to cloud servers—ensuring complete user privacy."

---

### [1:15 - 1:40] 4. Technical Architecture
**Visual on Screen**: Quickly show code in VS Code (`main.py`) or architecture diagram from `README.md`.  
**Speaker**:  
> "Under the hood, the application is built using Python and OpenCV.  
> 1. We capture the video stream using optimized DirectShow drivers to eliminate camera lag.  
> 2. We convert each frame to grayscale to reduce computing overhead by 66%.  
> 3. We use the pre-trained Viola-Jones Haar Cascade Classifier, leveraging integral images and attentional decision cascades for millisecond-level detection.  
> 4. We then render our bounding boxes and translucent HUD overlay."

---

### [1:40 - 2:00] 5. Future Scope & Conclusion
**Visual on Screen**: Back to the camera window. Press `Q` to show clean exit in terminal.  
**Speaker**:  
> "For future enhancements, we plan to integrate facial recognition for automated attendance logging and deploy this to edge devices like the Raspberry Pi.  
> By pressing 'Q', the system cleanly releases all hardware resources and closes.  
> Thank you HackZen 2.0 organizers and judges for this opportunity!"

---

### Quick Presentation Checklist Before Recording
- [ ] Good indoor lighting on your face (avoid bright light directly behind you).
- [ ] Close Zoom, Microsoft Teams, or Google Meet so the camera isn't locked.
- [ ] Run `python main.py` in full screen or cleanly sized terminal.
- [ ] Speak at a clear, confident pace.
- [ ] Have a second person or a smartphone photo with a face ready to demonstrate the multi-face count!
