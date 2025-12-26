import React, { useState, useEffect } from 'react';
import { getDrugs, getCancerTypes, predictSynergy } from '../api/api';
import DrugSelector from './DrugSelector';
import SynergyResult from './SynergyResult';
import SideEffects from './SideEffects';

function DrugCombination() {
  const [drugs, setDrugs] = useState([]);
  const [cancerTypes, setCancerTypes] = useState({});
  const [selectedDrugA, setSelectedDrugA] = useState('');
  const [selectedDrugB, setSelectedDrugB] = useState('');
  const [selectedCancerType, setSelectedCancerType] = useState('');
  const [selectedCellLine, setSelectedCellLine] = useState('');
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      console.log('Fetching data from backend...');
      
      const [drugsRes, cancerRes] = await Promise.all([
        getDrugs(),
        getCancerTypes()
      ]);
      
      console.log('Raw drugs response:', drugsRes);
      console.log('Raw cancer response:', cancerRes);
      
      // Try multiple ways to extract the data
      let drugsData = [];
      let cancerData = {};
      
      // Check different possible response structures
      if (drugsRes.data) {
        if (Array.isArray(drugsRes.data.drugs)) {
          drugsData = drugsRes.data.drugs;
        } else if (Array.isArray(drugsRes.data)) {
          drugsData = drugsRes.data;
        }
      }
      
      if (cancerRes.data) {
        if (cancerRes.data.cancer_types && typeof cancerRes.data.cancer_types === 'object') {
          cancerData = cancerRes.data.cancer_types;
        } else if (typeof cancerRes.data === 'object' && !Array.isArray(cancerRes.data)) {
          cancerData = cancerRes.data;
        }
      }
      
      console.log('Extracted drugs data:', drugsData);
      console.log('Extracted cancer data:', cancerData);
      console.log('Drugs array length:', drugsData.length);
      console.log('Cancer types count:', Object.keys(cancerData).length);
      
      if (!Array.isArray(drugsData) || drugsData.length === 0) {
        console.error('No drugs available. Response structure might be different.');
        console.log('Full drugsRes.data:', JSON.stringify(drugsRes.data, null, 2));
        setError('No drugs available from backend. Check console for details.');
        return;
      }
      
      if (!cancerData || Object.keys(cancerData).length === 0) {
        console.error('No cancer types available. Response structure might be different.');
        console.log('Full cancerRes.data:', JSON.stringify(cancerRes.data, null, 2));
        setError('No cancer types available from backend. Check console for details.');
        return;
      }
      
      // Set state - force re-render
      setDrugs([...drugsData]);
      setCancerTypes({...cancerData});
      
      console.log('State set. Drugs:', drugsData);
      console.log('State set. Cancer types:', cancerData);
      
      // Set default selections with a slight delay to ensure state is updated
      setTimeout(() => {
        if (drugsData.length > 0) {
          const firstDrug = drugsData[0].drug_name || drugsData[0];
          setSelectedDrugA(firstDrug);
          console.log('Set Drug A to:', firstDrug);
        }
        
        if (drugsData.length > 1) {
          const secondDrug = drugsData[1].drug_name || drugsData[1];
          setSelectedDrugB(secondDrug);
          console.log('Set Drug B to:', secondDrug);
        }
        
        const cancerTypeKeys = Object.keys(cancerData);
        if (cancerTypeKeys.length > 0) {
          const firstCancerType = cancerTypeKeys[0];
          setSelectedCancerType(firstCancerType);
          console.log('Set Cancer Type to:', firstCancerType);
          
          const cellLinesForType = cancerData[firstCancerType];
          if (Array.isArray(cellLinesForType) && cellLinesForType.length > 0) {
            setSelectedCellLine(cellLinesForType[0]);
            console.log('Set Cell Line to:', cellLinesForType[0]);
          }
        }
      }, 100);
      
      setError(null);
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Failed to load initial data';
      setError(`Backend Error: ${errorMessage}. Make sure Flask is running on port 5000.`);
      console.error('Error loading data:', err);
      console.error('Error details:', err.response);
    } finally {
      setLoading(false);
    }
  };

  // Log whenever drugs state changes
  useEffect(() => {
    console.log('Drugs state changed:', drugs);
    console.log('Drugs is array:', Array.isArray(drugs));
    console.log('Drugs length:', drugs.length);
  }, [drugs]);

  useEffect(() => {
    console.log('Cancer types state changed:', cancerTypes);
    console.log('Cancer types keys:', Object.keys(cancerTypes));
  }, [cancerTypes]);

  const handlePredict = async () => {
    if (!selectedDrugA || !selectedDrugB || !selectedCancerType || !selectedCellLine) {
      setError('Please select all required fields');
      return;
    }

    if (selectedDrugA === selectedDrugB) {
      setError('Please select two different drugs');
      return;
    }

    setPredicting(true);
    setError(null);
    setResult(null);

    try {
      console.log('Predicting with:', {
        drug_a: selectedDrugA,
        drug_b: selectedDrugB,
        cancer_type: selectedCancerType,
        cell_line: selectedCellLine
      });
      
      const response = await predictSynergy({
        drug_a: selectedDrugA,
        drug_b: selectedDrugB,
        cancer_type: selectedCancerType,
        cell_line: selectedCellLine
      });
      
      console.log('Prediction response:', response.data);
      setResult(response.data);
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Prediction failed';
      setError(errorMessage);
      console.error('Prediction error:', err);
    } finally {
      setPredicting(false);
    }
  };

  const cellLines = selectedCancerType ? (cancerTypes[selectedCancerType] || []) : [];

  // Show loading state while initial data is loading
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading data from backend...</p>
          <p className="text-gray-400 text-sm mt-2">Make sure Flask is running on port 5000</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-primary mb-4">
            🧪 Drug Combination Synergy Predictor
          </h1>
          <p className="text-gray-600 text-lg">
            Predict drug synergy and assess side effect risks using Graph Neural Networks
          </p>
          <div className="mt-2 text-sm text-gray-500">
            Loaded {drugs.length} drugs and {Object.keys(cancerTypes).length} cancer types
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-lg mb-8">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="font-bold">Error</p>
                <p>{error}</p>
                <button 
                  onClick={loadInitialData}
                  className="mt-2 text-sm underline hover:no-underline"
                >
                  Retry loading data
                </button>
              </div>
            </div>
          </div>
        )}



        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Left Column - Drug Selection */}
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800">Drug Selection</h2>
            <DrugSelector
              drugs={drugs}
              selectedDrug={selectedDrugA}
              onDrugChange={(value) => {
                console.log('Drug A changed to:', value);
                setSelectedDrugA(value);
              }}
              label="Drug A"
              moleculeImage={result?.drug_a?.image}
            />
            <DrugSelector
              drugs={drugs}
              selectedDrug={selectedDrugB}
              onDrugChange={(value) => {
                console.log('Drug B changed to:', value);
                setSelectedDrugB(value);
              }}
              label="Drug B"
              moleculeImage={result?.drug_b?.image}
            />
          </div>

          {/* Right Column - Cancer Context */}
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800">Cancer Context</h2>
            <div className="bg-white rounded-lg shadow-md p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Cancer Type ({Object.keys(cancerTypes).length} available)
                </label>
                <select
                  value={selectedCancerType}
                  onChange={(e) => {
                    const newType = e.target.value;
                    console.log('Cancer type changed to:', newType);
                    setSelectedCancerType(newType);
                    const newCellLines = cancerTypes[newType] || [];
                    setSelectedCellLine(newCellLines[0] || '');
                  }}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                  disabled={Object.keys(cancerTypes).length === 0}
                >
                  {Object.keys(cancerTypes).length === 0 ? (
                    <option value="">No cancer types available</option>
                  ) : (
                    Object.keys(cancerTypes).map((type) => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Cell Line ({cellLines.length} available)
                </label>
                <select
                  value={selectedCellLine}
                  onChange={(e) => {
                    console.log('Cell line changed to:', e.target.value);
                    setSelectedCellLine(e.target.value);
                  }}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                  disabled={cellLines.length === 0}
                >
                  {cellLines.length === 0 ? (
                    <option value="">No cell lines available</option>
                  ) : (
                    cellLines.map((line) => (
                      <option key={line} value={line}>
                        {line}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <button
                onClick={handlePredict}
                disabled={predicting || drugs.length === 0 || !selectedDrugA || !selectedDrugB}
                className="w-full bg-primary hover:bg-blue-600 text-white font-bold py-4 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
              >
                {predicting ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Predicting...
                  </span>
                ) : (
                  '🔬 Predict Synergy & Side Effects'
                )}
              </button>
              
              {/* Debug Info */}
              <div className="text-xs text-gray-500 mt-2 p-2 bg-gray-50 rounded">
                <p>Selected: {selectedDrugA || 'None'} + {selectedDrugB || 'None'}</p>
                <p>Cancer: {selectedCancerType || 'None'} / {selectedCellLine || 'None'}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-8">
            <SynergyResult result={result} />
            
            {result.side_effects_a && result.side_effects_b && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <SideEffects
                  sideEffects={result.side_effects_a}
                  title={`💊 ${result.drug_a.name} Side Effect Profile`}
                />
                <SideEffects
                  sideEffects={result.side_effects_b}
                  title={`💊 ${result.drug_b.name} Side Effect Profile`}
                />
              </div>
            )}
            
            {result.combination_side_effects && (
              <SideEffects
                sideEffects={result.combination_side_effects}
                title="🔄 Combination Side Effect Risks"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default DrugCombination;
