import React from 'react';

const SideEffects = ({ sideEffects, title }) => {
  const getRiskLevel = (score) => {
    if (score >= 0.7) return { level: 'High', color: 'bg-red-100 border-red-400 text-red-800' };
    if (score >= 0.4) return { level: 'Medium', color: 'bg-yellow-100 border-yellow-400 text-yellow-800' };
    return { level: 'Low', color: 'bg-green-100 border-green-400 text-green-800' };
  };

  const formatEffectName = (name) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const groupedEffects = {
    high: [],
    medium: [],
    low: []
  };

  Object.entries(sideEffects).forEach(([effect, score]) => {
    const { level, color } = getRiskLevel(score);
    const item = { effect, score, level, color };
    
    if (level === 'High') groupedEffects.high.push(item);
    else if (level === 'Medium') groupedEffects.medium.push(item);
    else groupedEffects.low.push(item);
  });

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-bold text-gray-800 mb-4">{title}</h3>
      
      <div className="space-y-4">
        {groupedEffects.high.length > 0 && (
          <div>
            <p className="text-sm font-semibold text-red-600 mb-2">🚨 High Risk (0.7-1.0)</p>
            <div className="space-y-2">
              {groupedEffects.high.map(({ effect, score, color }) => (
                <div key={effect} className={`p-3 rounded-lg border-l-4 ${color}`}>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{formatEffectName(effect)}</span>
                    <span className="font-bold">{score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {groupedEffects.medium.length > 0 && (
          <div>
            <p className="text-sm font-semibold text-yellow-600 mb-2">⚠️ Medium Risk (0.4-0.7)</p>
            <div className="space-y-2">
              {groupedEffects.medium.map(({ effect, score, color }) => (
                <div key={effect} className={`p-3 rounded-lg border-l-4 ${color}`}>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{formatEffectName(effect)}</span>
                    <span className="font-bold">{score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {groupedEffects.low.length > 0 && (
          <div>
            <p className="text-sm font-semibold text-green-600 mb-2">✅ Low Risk (0.0-0.4)</p>
            <div className="space-y-2">
              {groupedEffects.low.map(({ effect, score, color }) => (
                <div key={effect} className={`p-3 rounded-lg border-l-4 ${color}`}>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{formatEffectName(effect)}</span>
                    <span className="font-bold">{score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SideEffects;
