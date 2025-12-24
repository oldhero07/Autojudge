export interface PredictionRequest {
  description: string;
  input_desc: string;
  output_desc: string;
}

export interface PredictionResponse {
  class: string;
  score: number;
  features?: {
    textLength: number;
    mathSymbols: number;
    keywords: number;
    tfidfFeatures: number;
  };
}

export interface TextFeatures {
  length: number;
  mathSymbols: number;
  keywords: number;
  tfidfScore: number;
}

export interface ApiError {
  message: string;
}
