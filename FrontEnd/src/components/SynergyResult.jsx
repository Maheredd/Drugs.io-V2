import React from 'react';
import { FaCheckCircle, FaExclamationTriangle, FaTimesCircle } from 'react-icons/fa';

const SynergyResult = ({ result }) => {
  const getSynergyColor = (score) => {
    if (score > 10) return 'text-success';
    if (score > 5) return 'text-warning';
    if (score > 0) return 'text-primary';
    return 'text-danger';
  };

  const getSynergyIcon = (score) => {
    if (score > 5) return <FaCheckCircle className="inline mr-2" />;
    if (score > 0) return <FaExclamationTriangle className="inline mr-2" />;
    return <FaTimesCircle className="inline mr-2" />;
  };

  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg shadow-lg p-8 border-l-4 border-primary">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        🎯 Synergy Prediction Results
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <p className="text-sm text-gray-600 mb-2">Synergy Score</p>
          <p className={`text-4xl font-bold ${getSynergyColor(result.synergy_score)}`}>
            {result.synergy_score.toFixed(2)}
          </p>
        </div>
        
        <div>
          <p className="text-sm text-gray-600 mb-2">Synergy Class</p>
          <p className={`text-2xl font-bold ${getSynergyColor(result.synergy_score)}`}>
            {getSynergyIcon(result.synergy_score)}
            {result.synergy_class}
          </p>
        </div>
      </div>
      
      <div className="mt-6 space-y-3">
        <div className="bg-white rounded-lg p-4">
          <p className="text-sm font-medium text-gray-700">Interpretation</p>
          <p className="text-gray-900 mt-1">{result.interpretation}</p>
        </div>
        
        <div className="bg-white rounded-lg p-4">
          <p className="text-sm font-medium text-gray-700">Recommendation</p>
          <p className="text-gray-900 mt-1">{result.recommendation}</p>
        </div>
      </div>
    </div>
  );
};

export default SynergyResult;
