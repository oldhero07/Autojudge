import React from 'react';
import { PredictionResponse } from '../types';
import { CheckCircle2, BarChart3, Hash, Calculator, Tag, FileText } from 'lucide-react';

interface ResultDisplayProps {
  result: PredictionResponse;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  // Get difficulty class based on score for consistency
  const getDifficultyFromScore = (score: number) => {
    if (score < 4) return 'easy';
    if (score < 7) return 'medium';
    return 'hard';
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20';
      case 'medium': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'hard': return 'text-rose-400 bg-rose-400/10 border-rose-400/20';
      default: return 'text-slate-400 bg-slate-400/10 border-slate-400/20';
    }
  };

  const getScoreColor = (score: number) => {
    if (score < 4) return 'text-emerald-400';    // Easy: 1-4
    if (score < 7) return 'text-yellow-400';     // Medium: 4-7  
    return 'text-rose-400';                      // Hard: 7-10
  };

  // Use score-based difficulty for consistency, but show ML prediction in parentheses if different
  const scoreDifficulty = getDifficultyFromScore(result.score);
  const mlDifficulty = result.class.toLowerCase();
  const difficultyClass = getDifficultyColor(scoreDifficulty);

  return (
    <div 
      id="result-container" 
      className="mt-8 p-6 bg-slate-950/50 rounded-xl border border-slate-700 animate-in fade-in slide-in-from-bottom-4 duration-500"
    >
      <div className="flex items-center gap-2 mb-4">
        <CheckCircle2 className="w-5 h-5 text-indigo-400" />
        <h3 className="text-lg font-semibold text-slate-200">Analysis Result</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Difficulty Class Card */}
        <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 flex flex-col items-center justify-center gap-2">
          <span className="text-sm text-slate-500 uppercase tracking-wider font-semibold">Difficulty</span>
          <span 
            id="result-class" 
            className={`text-2xl font-bold px-4 py-1 rounded-full border ${difficultyClass}`}
          >
            {scoreDifficulty}
            {mlDifficulty !== scoreDifficulty && (
              <span className="text-xs opacity-70 ml-1">({result.class})</span>
            )}
          </span>
        </div>

        {/* Score Card */}
        <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 flex flex-col items-center justify-center gap-2">
          <span className="text-sm text-slate-500 uppercase tracking-wider font-semibold flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Complexity Score
          </span>
          <div className="flex items-baseline gap-1">
            <span id="result-score" className={`text-3xl font-bold ${getScoreColor(result.score)}`}>
              {result.score}
            </span>
            <span className="text-sm text-slate-600">/ 10</span>
          </div>
        </div>
      </div>

      {/* Feature Analysis */}
      {result.features && (
        <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-800">
          <h4 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
            <Calculator className="w-4 h-4 text-indigo-400" />
            ML Feature Analysis
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex flex-col items-center p-3 bg-slate-800/50 rounded-lg">
              <FileText className="w-4 h-4 text-slate-400 mb-1" />
              <span className="text-slate-400 text-xs">Text Length</span>
              <span className="text-indigo-300 font-mono text-lg">{result.features.textLength.toLocaleString()}</span>
            </div>
            <div className="flex flex-col items-center p-3 bg-slate-800/50 rounded-lg">
              <Hash className="w-4 h-4 text-slate-400 mb-1" />
              <span className="text-slate-400 text-xs">Math Symbols</span>
              <span className="text-indigo-300 font-mono text-lg">{result.features.mathSymbols}</span>
            </div>
            <div className="flex flex-col items-center p-3 bg-slate-800/50 rounded-lg">
              <Tag className="w-4 h-4 text-slate-400 mb-1" />
              <span className="text-slate-400 text-xs">Keywords</span>
              <span className="text-indigo-300 font-mono text-lg">{result.features.keywords}</span>
            </div>
            <div className="flex flex-col items-center p-3 bg-slate-800/50 rounded-lg">
              <BarChart3 className="w-4 h-4 text-slate-400 mb-1" />
              <span className="text-slate-400 text-xs">TF-IDF Features</span>
              <span className="text-indigo-300 font-mono text-lg">{result.features.tfidfFeatures}</span>
            </div>
          </div>
          
          {/* Scoring explanation */}
          <div className="mt-4 p-3 bg-slate-800/30 rounded-lg border border-slate-700">
            <p className="text-xs text-slate-400">
              <strong>Scoring Guide:</strong> 1-4 (Easy), 4-7 (Medium), 7-10 (Hard) • 
              Codeforces ~2000 rating ≈ 6-7/10 • 
              Display shows score-based difficulty for consistency
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
