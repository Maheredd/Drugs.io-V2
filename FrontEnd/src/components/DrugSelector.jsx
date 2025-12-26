import React from 'react';

const DrugSelector = ({ drugs = [], selectedDrug, onDrugChange, label, moleculeImage }) => {
  // console.log('DrugSelector render:', { drugs, selectedDrug, label });
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      <select
        value={selectedDrug || ''}
        onChange={(e) => onDrugChange(e.target.value)}
        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
        disabled={!drugs || drugs.length === 0}
      >
        <option value="">Select a drug</option>
        {Array.isArray(drugs) && drugs.map((drug) => (
          <option key={drug.drug_name || drug} value={drug.drug_name || drug}>
            {drug.drug_name || drug}
          </option>
        ))}
      </select>
      
      {drugs.length === 0 && (
        <p className="text-sm text-red-500 mt-2">
          No drugs loaded. Check backend connection.
        </p>
      )}
      
      {moleculeImage && (
        <div className="mt-4 flex justify-center">
          <img
            src={moleculeImage}
            alt={`${selectedDrug} structure`}
            className="rounded-lg border-2 border-gray-200 max-w-full h-auto"
          />
        </div>
      )}
      
      {selectedDrug && !moleculeImage && (
        <div className="mt-4 flex justify-center">
          <div className="text-gray-400 text-sm">
            Molecule structure will appear after prediction
          </div>
        </div>
      )}
    </div>
  );
};

export default DrugSelector;
