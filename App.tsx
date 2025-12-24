import React, { useState } from 'react';
import { Brain, Code2, Loader2, Sparkles, AlertTriangle, ToggleLeft, ToggleRight } from 'lucide-react';
import { InputGroup } from './components/InputGroup';
import { ResultDisplay } from './components/ResultDisplay';
import { PredictionResponse, PredictionRequest } from './types';

const App: React.FC = () => {
  // Input mode state
  const [useThreeInputs, setUseThreeInputs] = useState(true); // Default to three inputs as per AutoJudge spec
  
  // Three separate inputs (AutoJudge format)
  const [problemDescription, setProblemDescription] = useState('');
  const [inputDescription, setInputDescription] = useState('');
  const [outputDescription, setOutputDescription] = useState('');
  
  // Legacy combined input (backward compatibility)
  const [combinedText, setCombinedText] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [textFeatures, setTextFeatures] = useState({
    length: 0,
    mathSymbols: 0,
    keywords: 0,
    tfidfScore: 0
  });

  // Get combined text for feature analysis
  const getCombinedText = () => {
    if (useThreeInputs) {
      return `${problemDescription} ${inputDescription} ${outputDescription}`.trim();
    }
    return combinedText;
  };

  // Feature extraction functions
  const extractFeatures = (text: string) => {
    // Text length
    const length = text.length;
    
    // Enhanced mathematical symbols count (more comprehensive)
    const mathSymbolsRegex = /[+\-*/=<>$^∑∏∫∂∆√π∞≤≥≠≈∈∉∪∩⊂⊃∅%&|~!@#()[\]{}]/g;
    const mathSymbols = (text.match(mathSymbolsRegex) || []).length;
    
    // Enhanced keyword frequency for difficulty indicators
    const difficultyKeywords = [
      'graph', 'dp', 'recursion', 'tree', 'constraint', 'query', 
      'algorithm', 'optimize', 'complexity', 'binary', 'search',
      'sort', 'heap', 'stack', 'queue', 'dynamic', 'greedy',
      'backtrack', 'dfs', 'bfs', 'shortest', 'path', 'minimum',
      'maximum', 'optimal', 'efficient'
    ];
    const textLower = text.toLowerCase();
    const keywords = difficultyKeywords.reduce((count, keyword) => {
      return count + (textLower.match(new RegExp(`\\b${keyword}\\b`, 'g')) || []).length;
    }, 0);
    
    // Enhanced TF-IDF approximation with better word boundary detection
    const words = text.toLowerCase().match(/\b[a-zA-Z]{2,}\b/g) || [];
    const uniqueWords = new Set(words);
    const tfidfScore = words.length > 0 ? uniqueWords.size / words.length : 0;
    
    return { length, mathSymbols, keywords, tfidfScore };
  };

  // Update features when any text changes
  const updateFeatures = () => {
    const currentText = getCombinedText();
    if (currentText.length > 0) {
      setTextFeatures(extractFeatures(currentText));
    } else {
      setTextFeatures({ length: 0, mathSymbols: 0, keywords: 0, tfidfScore: 0 });
    }
  };

  // Handle three-input changes
  const handleProblemDescriptionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setProblemDescription(e.target.value);
    setTimeout(updateFeatures, 0); // Update features after state update
  };

  const handleInputDescriptionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputDescription(e.target.value);
    setTimeout(updateFeatures, 0);
  };

  const handleOutputDescriptionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setOutputDescription(e.target.value);
    setTimeout(updateFeatures, 0);
  };

  // Handle legacy combined text change
  const handleCombinedTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCombinedText(e.target.value);
    setTimeout(updateFeatures, 0);
  };

  // Toggle between input modes
  const handleInputModeToggle = () => {
    setUseThreeInputs(!useThreeInputs);
    // Clear results when switching modes
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    const currentText = getCombinedText();
    
    // Validation
    if (!currentText.trim()) {
      setError("Please enter the problem description to generate a prediction.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    // Prepare payload based on input mode
    const payload: PredictionRequest = useThreeInputs 
      ? {
          description: problemDescription,
          input_desc: inputDescription,
          output_desc: outputDescription,
        }
      : {
          description: combinedText,
          input_desc: "",
          output_desc: "",
        };

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error("Prediction failed:", err);
      setError(err.message || "Failed to connect to prediction server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Header Section */}
      <div className="text-center mb-10 space-y-2">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="p-3 bg-indigo-500/10 rounded-xl border border-indigo-500/20">
            <Brain className="w-8 h-8 text-indigo-400" />
          </div>
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-300 tracking-tight">
            AutoJudge
          </h1>
        </div>
        <p className="text-slate-400 text-lg font-light tracking-wide">
          Programming Problem Difficulty Predictor
        </p>
      </div>

      {/* Main Card */}
      <div className="w-full max-w-3xl bg-slate-900/50 backdrop-blur-sm rounded-2xl border border-slate-800 shadow-2xl overflow-hidden p-1">
        <div className="bg-slate-900/80 p-6 sm:p-8 rounded-xl space-y-6">
          
          {/* Input Mode Toggle */}
          <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-lg border border-slate-700">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
                <Code2 className="w-4 h-4 text-indigo-400" />
              </div>
              <div>
                <h3 className="text-sm font-medium text-slate-300">Input Format</h3>
                <p className="text-xs text-slate-500">
                  {useThreeInputs ? 'AutoJudge Research Format (3 separate fields)' : 'Legacy Combined Format'}
                </p>
              </div>
            </div>
            <button
              onClick={handleInputModeToggle}
              className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
            >
              {useThreeInputs ? (
                <>
                  <ToggleRight className="w-4 h-4 text-indigo-400" />
                  <span className="text-sm text-slate-300">Three Inputs</span>
                </>
              ) : (
                <>
                  <ToggleLeft className="w-4 h-4 text-slate-500" />
                  <span className="text-sm text-slate-300">Combined</span>
                </>
              )}
            </button>
          </div>

          {/* Input Fields */}
          <div className="space-y-6">
            {useThreeInputs ? (
              // Three separate inputs (AutoJudge format)
              <>
                <InputGroup
                  id="problem_description"
                  label="Problem Description"
                  value={problemDescription}
                  onChange={handleProblemDescriptionChange}
                  placeholder="Enter the main problem statement, including the task description, constraints, and any background information..."
                  height="h-40"
                  icon={<Code2 className="w-4 h-4" />}
                />
                
                <InputGroup
                  id="input_description"
                  label="Input Description"
                  value={inputDescription}
                  onChange={handleInputDescriptionChange}
                  placeholder="Describe the input format, including the number of test cases, data types, ranges, and structure..."
                  height="h-24"
                  icon={<Code2 className="w-4 h-4" />}
                />
                
                <InputGroup
                  id="output_description"
                  label="Output Description"
                  value={outputDescription}
                  onChange={handleOutputDescriptionChange}
                  placeholder="Describe the expected output format, including what should be printed and any formatting requirements..."
                  height="h-24"
                  icon={<Code2 className="w-4 h-4" />}
                />
              </>
            ) : (
              // Legacy combined input
              <InputGroup
                id="combined_text"
                label="Problem Description (Combined)"
                value={combinedText}
                onChange={handleCombinedTextChange}
                placeholder="Enter the complete problem statement including description, input format, output format, constraints, and examples..."
                height="h-64"
                icon={<Code2 className="w-4 h-4" />}
              />
            )}

            {/* Feature Display */}
            {getCombinedText() && (
              <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700">
                <h3 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  Text Analysis Features
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="flex flex-col">
                    <span className="text-slate-400">Text Length</span>
                    <span className="text-indigo-300 font-mono">{textFeatures.length.toLocaleString()}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-slate-400">Math Symbols</span>
                    <span className="text-indigo-300 font-mono">{textFeatures.mathSymbols}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-slate-400">Keywords</span>
                    <span className="text-indigo-300 font-mono">{textFeatures.keywords}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-slate-400">Vocabulary Ratio</span>
                    <span className="text-indigo-300 font-mono">{textFeatures.tfidfScore.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Action Button */}
          <div className="pt-4">
            <button
              id="predict-btn"
              onClick={handlePredict}
              disabled={loading}
              className={`
                w-full py-4 px-6 rounded-lg flex items-center justify-center gap-3
                text-white font-semibold text-lg shadow-lg
                transition-all duration-300 transform active:scale-[0.98]
                ${loading 
                  ? 'bg-slate-800 cursor-not-allowed text-slate-400' 
                  : 'bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 hover:shadow-indigo-500/25 ring-1 ring-white/10'
                }
              `}
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing Complexity...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 text-indigo-200" />
                  <span>Predict Difficulty</span>
                </>
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-rose-950/30 border border-rose-900/50 rounded-lg flex items-start gap-3 text-rose-300 animate-in fade-in slide-in-from-top-2">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Results */}
          {result && <ResultDisplay result={result} />}

        </div>
      </div>
      
      {/* Footer */}
      <footer className="mt-12 text-slate-600 text-sm">
        <p>&copy; {new Date().getFullYear()} AutoJudge System. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default App;
