# AutoJudge Demo Video Script
## Duration: 3-5 minutes

### Scene 1: Introduction (30 seconds)
**[Screen: GitHub repository homepage]**

"Hello! I'm presenting AutoJudge, a machine learning system that automatically predicts programming problem difficulty. This system analyzes problem descriptions and classifies them as Easy, Medium, or Hard, while also providing numerical difficulty scores from 1 to 10."

**[Screen: README.md showing performance metrics]**

"The system achieves 55% classification accuracy on over 4,000 programming problems and provides real-time predictions through a web interface."

### Scene 2: Problem Demonstration (45 seconds)
**[Screen: Open web interface at localhost:5000]**

"Let me demonstrate how AutoJudge works. Here's the web interface where users can input programming problem descriptions."

**[Type in text area]**
"I'll enter a simple problem: 'Print Hello World to the console'"

**[Click Predict button]**

"As expected, AutoJudge classifies this as 'Easy' with a score of 2.1 out of 10, showing high confidence at 87%."

**[Clear and enter new problem]**
"Now let's try a more complex problem: 'Find the shortest path in a weighted graph using Dijkstra's algorithm with binary heap optimization'"

**[Click Predict button]**

"AutoJudge correctly identifies this as 'Medium' difficulty with a score of 6.8 out of 10."

### Scene 3: Technical Architecture (60 seconds)
**[Screen: Show PROJECT_STRUCTURE.md or code]**

"Let me show you the technical implementation. AutoJudge uses a sophisticated machine learning pipeline:"

**[Screen: Show flask_app/app.py - feature extraction section]**

"First, it extracts 15 custom features from the problem text, including algorithm indicators, complexity markers, and text metrics."

**[Screen: Show TF-IDF vectorization code]**

"Then it applies TF-IDF vectorization to create 3,000 text features, combined with the custom features for a total of 3,015 features."

**[Screen: Show model architecture]**

"The classification uses an ensemble of three models: Logistic Regression, Random Forest, and Gradient Boosting, combined through soft voting for robust predictions."

### Scene 4: API Demonstration (45 seconds)
**[Screen: Terminal or Postman]**

"AutoJudge also provides a REST API for programmatic access. Let me demonstrate using curl:"

**[Type command]**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Implement binary search on a sorted array"}'
```

**[Show JSON response]**
"The API returns detailed predictions including confidence scores and feature analysis, making it easy to integrate into other systems."

### Scene 5: Performance Analysis (30 seconds)
**[Screen: Show confusion matrix and metrics]**

"Let's look at the performance metrics. On a test set of 823 problems, AutoJudge achieves:"
- "55% overall accuracy"
- "Best performance on Hard problems with 66% F1-score"
- "Mean Absolute Error of 1.7 points for score prediction"

**[Screen: Show per-class performance table]**

"The system performs well across all difficulty levels, with particularly strong results for identifying hard problems."

### Scene 6: Deployment and Production (30 seconds)
**[Screen: Show Docker files and deployment documentation]**

"AutoJudge is production-ready with Docker containerization, comprehensive documentation, and deployment guides for cloud platforms."

**[Screen: Show GitHub repository structure]**

"The complete source code, documentation, and trained models are available on GitHub, making it easy to deploy and extend."

### Scene 7: Conclusion (20 seconds)
**[Screen: Return to GitHub repository]**

"AutoJudge demonstrates practical application of machine learning in educational technology. It provides automated difficulty assessment for programming problems, helping educators, competitive programming platforms, and coding interview services."

**[Screen: Show README with contact/contribution info]**

"Thank you for watching! The project is open source and contributions are welcome. All documentation and deployment instructions are available in the repository."

---

## 🎬 Demo Video Recording Checklist

### Pre-Recording Setup:
- [ ] Start Flask application (`cd flask_app && python app.py`)
- [ ] Open web browser to `http://localhost:5000`
- [ ] Prepare terminal with curl commands
- [ ] Have GitHub repository open in another tab
- [ ] Test all demo scenarios beforehand

### Recording Tips:
- [ ] Use screen recording software (OBS, Camtasia, or built-in tools)
- [ ] Record in 1080p resolution
- [ ] Ensure clear audio with minimal background noise
- [ ] Keep cursor movements smooth and deliberate
- [ ] Pause briefly between sections for editing
- [ ] Record 10-15 seconds extra at beginning and end

### Post-Recording:
- [ ] Edit for smooth transitions
- [ ] Add captions if required
- [ ] Export in common format (MP4, MOV)
- [ ] Keep file size reasonable (< 100MB if possible)
- [ ] Upload to repository or provide download link

### Alternative: Screenshots + Narration
If video recording is challenging:
- [ ] Take high-quality screenshots of each step
- [ ] Create slide presentation with screenshots
- [ ] Record narration over slides
- [ ] Export as video or interactive presentation