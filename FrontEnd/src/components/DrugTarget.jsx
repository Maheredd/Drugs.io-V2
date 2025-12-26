import { useState } from 'react';
import { 
  ArrowRight, 
  TestTube, 
  Dna, 
  Brain, 
  ChartLine, 
  Sparkles, 
  AlertTriangle, 
  Microscope, 
  Circle,
  Zap,
  TrendingUp,
  CheckCircle2,
  XCircle,
  Info,
  Download,
  RefreshCw,
  Beaker
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function DrugTarget() {
  const [smiles, setSmiles] = useState('');
  const [protein, setProtein] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('predictor');

  const examples = {
    aspirin: {
      smiles: 'CC(=O)Oc1ccccc1C(=O)O',
      protein: 'MAGEKIVKFKELLEQKAETSNGVLLDLACQEPQQHYLKLDRRLENSEAYVAKKQSDMAGWSLYDCLDYDELPTQVDYQWRRAARDAAKTTLAQMTFAAI',
      name: 'Aspirin + COX'
    },
    hiv: {
      smiles: 'CC1C(=O)NC(=O)NC1=O',
      protein: 'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGKWKYVGRGGPSVGEVLERLAKLAGNV',
      name: 'HIV Protease Inhibitor'
    },
    cancer: {
      smiles: 'CC1=CNc2c1c(OC3CCNCC3)nc(n2)N',
      protein: 'MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQGVPI',
      name: 'Cancer Kinase Inhibitor'
    }
  };

  const loadExample = (exampleKey) => {
    setSmiles(examples[exampleKey].smiles);
    setProtein(examples[exampleKey].protein);
    setError('');
    setResults(null);
  };

  const handlePredict = async () => {
    if (!smiles.trim() || !protein.trim()) {
      setError('Both SMILES and protein sequence are required.');
      return;
    }

    setError('');
    setResults(null);
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5002/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles, protein_sequence: protein })
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResults(data);
      }
    } catch (err) {
      setError('Failed to connect to the server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSmiles('');
    setProtein('');
    setResults(null);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-purple-900/20 dark:to-indigo-900/20">
      {/* Enhanced Hero Section */}
      <motion.section 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative overflow-hidden"
      >
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              rotate: [0, 180, 360],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute -top-24 -right-24 w-96 h-96 bg-purple-300/20 dark:bg-purple-500/10 rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1.2, 1, 1.2],
              rotate: [360, 180, 0],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute -bottom-24 -left-24 w-96 h-96 bg-indigo-300/20 dark:bg-indigo-500/10 rounded-full blur-3xl"
          />
        </div>

        <div className="relative text-center py-12 md:py-20 px-4">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="inline-flex items-center gap-2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm px-6 py-2 rounded-full mb-6 shadow-lg"
          >
            <Zap className="w-4 h-4 text-yellow-500" />
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Powered by Deep Learning
            </span>
          </motion.div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold mb-6 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 dark:from-indigo-400 dark:via-purple-400 dark:to-pink-400 bg-clip-text text-transparent leading-tight">
            Drug-Target Interaction
            <br />
            <span className="text-3xl md:text-5xl lg:text-6xl">Prediction Platform</span>
          </h1>

          <p className="text-lg md:text-xl mb-8 text-gray-700 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Accelerate pharmaceutical research with AI-powered binding affinity prediction.
            <br className="hidden md:block" />
            Get instant insights into drug-protein interactions with confidence scores.
          </p>

          <div className="flex flex-wrap gap-4 justify-center items-center">
            <motion.a
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              href="#predictor"
              className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-xl hover:shadow-2xl transition-all duration-300"
            >
              <Sparkles className="w-5 h-5" />
              Start Prediction
              <ArrowRight className="w-5 h-5" />
            </motion.a>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveTab('info')}
              className="inline-flex items-center gap-2 bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm text-gray-700 dark:text-gray-300 px-8 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 border-2 border-gray-200 dark:border-gray-700"
            >
              <Info className="w-5 h-5" />
              How It Works
            </motion.button>
          </div>
        </div>
      </motion.section>

      {/* How It Works - Redesigned */}
      {activeTab === 'info' && (
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-md py-16 px-4 border-y border-gray-200 dark:border-gray-700"
        >
          <div className="max-w-7xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 text-gray-900 dark:text-white">
              How It Works
            </h2>
            <p className="text-center text-gray-600 dark:text-gray-400 mb-12 max-w-2xl mx-auto">
              Our AI model analyzes molecular structures and protein sequences to predict binding interactions
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { 
                  icon: TestTube, 
                  title: 'Input Drug', 
                  desc: 'Provide SMILES notation of your compound',
                  color: 'from-blue-500 to-cyan-500',
                  delay: 0.1
                },
                { 
                  icon: Dna, 
                  title: 'Input Protein', 
                  desc: 'Enter the target protein amino acid sequence',
                  color: 'from-purple-500 to-pink-500',
                  delay: 0.2
                },
                { 
                  icon: Brain, 
                  title: 'AI Analysis', 
                  desc: 'Neural network processes molecular features',
                  color: 'from-orange-500 to-red-500',
                  delay: 0.3
                },
                { 
                  icon: TrendingUp, 
                  title: 'Get Insights', 
                  desc: 'Receive detailed binding probability and metrics',
                  color: 'from-green-500 to-emerald-500',
                  delay: 0.4
                }
              ].map((step, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: step.delay }}
                  whileHover={{ y: -8, scale: 1.02 }}
                  className="relative group"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-white/50 to-gray-100/50 dark:from-gray-800/50 dark:to-gray-900/50 rounded-2xl blur-xl group-hover:blur-2xl transition-all duration-300" />
                  <div className="relative bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-gray-200 dark:border-gray-700 h-full">
                    <div className={`w-16 h-16 mx-auto mb-5 bg-gradient-to-br ${step.color} rounded-2xl flex items-center justify-center text-white shadow-lg transform group-hover:rotate-6 transition-transform duration-300`}>
                      <step.icon className="w-8 h-8" />
                    </div>
                    <div className="text-center mb-2 text-gray-500 dark:text-gray-400 font-bold text-sm">
                      STEP {idx + 1}
                    </div>
                    <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white text-center">
                      {step.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 text-center leading-relaxed">
                      {step.desc}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="mt-12 text-center"
            >
              <button
                onClick={() => setActiveTab('predictor')}
                className="inline-flex items-center gap-2 text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300 font-semibold"
              >
                <ArrowRight className="w-5 h-5" />
                Go to Predictor
              </button>
            </motion.div>
          </div>
        </motion.section>
      )}

      {/* Main Predictor Section - Enhanced */}
      <section id="predictor" className="pb-16 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-10"
          >
            <p className="text-[38px] text-gray-600 dark:text-gray-400">
              Input your data or try our example compounds
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Input Panel - Enhanced */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-3xl blur-xl group-hover:blur-2xl transition-all duration-300" />
              <div className="relative bg-white/95 dark:bg-gray-800/95 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-3 mb-8">
                  <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl">
                    <Beaker className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Input Panel
                  </h3>
                </div>

                <div className="space-y-6">
                  {/* SMILES Input */}
                  <div className="space-y-3">
                    <label className="flex items-center gap-2 font-semibold text-gray-800 dark:text-gray-200">
                      <TestTube className="w-5 h-5 text-indigo-500" />
                      Drug SMILES String
                      <span className="group relative">
                        <Info className="w-4 h-4 text-gray-400 hover:text-indigo-500 cursor-help transition-colors" />
                        <span className="absolute left-6 top-0 w-64 p-3 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity shadow-xl z-10">
                          Simplified Molecular Input Line Entry System - A notation for representing chemical structures
                        </span>
                      </span>
                    </label>
                    <textarea
                      value={smiles}
                      onChange={(e) => setSmiles(e.target.value)}
                      rows={3}
                      placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O"
                      className="w-full px-4 py-3 border-2 border-gray-300 dark:border-gray-600 rounded-xl font-mono text-sm resize-none focus:outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/20 transition-all bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                      <Circle className="w-2 h-2 fill-current" />
                      Paste the SMILES notation of your compound
                    </p>
                  </div>

                  {/* Protein Input */}
                  <div className="space-y-3">
                    <label className="flex items-center gap-2 font-semibold text-gray-800 dark:text-gray-200">
                      <Dna className="w-5 h-5 text-purple-500" />
                      Target Protein Sequence
                      <span className="group relative">
                        <Info className="w-4 h-4 text-gray-400 hover:text-purple-500 cursor-help transition-colors" />
                        <span className="absolute left-6 top-0 w-64 p-3 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity shadow-xl z-10">
                          Amino acid sequence in single-letter code format (e.g., MKVLWALLVTF...)
                        </span>
                      </span>
                    </label>
                    <textarea
                      value={protein}
                      onChange={(e) => setProtein(e.target.value)}
                      rows={4}
                      placeholder="e.g., MKVLWALLVTFLAGCQAKVE..."
                      className="w-full px-4 py-3 border-2 border-gray-300 dark:border-gray-600 rounded-xl font-mono text-sm resize-none focus:outline-none focus:border-purple-500 focus:ring-4 focus:ring-purple-500/20 transition-all bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                      <Circle className="w-2 h-2 fill-current" />
                      Enter amino acid sequence in single-letter format
                    </p>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3 pt-4">
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handlePredict}
                      disabled={loading}
                      className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white py-4 rounded-xl font-semibold text-lg flex items-center justify-center gap-2 shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                    >
                      {loading ? (
                        <>
                          <RefreshCw className="w-5 h-5 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-5 h-5" />
                          Predict Interaction
                        </>
                      )}
                    </motion.button>

                    {(smiles || protein || results) && (
                      <motion.button
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleReset}
                        className="px-6 py-4 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-xl font-semibold transition-all duration-300 flex items-center gap-2"
                      >
                        <RefreshCw className="w-5 h-5" />
                        Reset
                      </motion.button>
                    )}
                  </div>

                  {/* Example Pills */}
                  <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                    <p className="font-semibold mb-3 text-gray-700 dark:text-gray-300 text-center flex items-center justify-center gap-2">
                      <TestTube className="w-4 h-4" />
                      Quick Examples
                    </p>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {Object.entries(examples).map(([key, example]) => (
                        <motion.button
                          key={key}
                          whileHover={{ scale: 1.05, y: -2 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => loadExample(key)}
                          className="px-5 py-2.5 bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-600 hover:from-indigo-100 hover:to-purple-100 dark:hover:from-indigo-900/50 dark:hover:to-purple-900/50 text-gray-700 dark:text-gray-300 rounded-full text-sm font-medium transition-all duration-200 shadow-md hover:shadow-lg border border-gray-300 dark:border-gray-600"
                        >
                          {example.name}
                        </motion.button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Results Panel - Enhanced */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-3xl blur-xl group-hover:blur-2xl transition-all duration-300" />
              <div className="relative bg-white/95 dark:bg-gray-800/95 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-gray-200 dark:border-gray-700 min-h-[600px] flex flex-col">
                <AnimatePresence mode="wait">
                  {loading && (
                    <motion.div
                      key="loading"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center"
                    >
                      <motion.div
                        animate={{
                          rotate: 360,
                          scale: [1, 1.1, 1]
                        }}
                        transition={{
                          rotate: { duration: 2, repeat: Infinity, ease: "linear" },
                          scale: { duration: 1, repeat: Infinity }
                        }}
                        className="w-20 h-20 mb-6 border-8 border-gray-200 dark:border-gray-700 border-t-indigo-500 rounded-full"
                      />
                      <p className="text-gray-600 dark:text-gray-400 font-medium text-lg">
                        Analyzing molecular interactions...
                      </p>
                      <p className="text-gray-500 dark:text-gray-500 text-sm mt-2">
                        This may take a few seconds
                      </p>
                    </motion.div>
                  )}

                  {error && (
                    <motion.div
                      key="error"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      className="flex-1 flex items-center justify-center"
                    >
                      <div className="bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800 rounded-2xl p-6 flex flex-col items-center gap-4 max-w-md">
                        <div className="w-16 h-16 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center">
                          <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
                        </div>
                        <div className="text-center">
                          <h4 className="font-bold text-red-800 dark:text-red-300 text-lg mb-2">
                            Prediction Error
                          </h4>
                          <p className="text-red-700 dark:text-red-400 text-sm leading-relaxed">
                            {error}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {results && !loading && !error && (
                    <motion.div
                      key="results"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      className="flex-1"
                    >
                      <ResultsDisplay results={results} />
                    </motion.div>
                  )}

                  {!loading && !error && !results && (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center text-center"
                    >
                      <motion.div
                        animate={{
                          y: [0, -10, 0]
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          ease: "easeInOut"
                        }}
                        className="w-24 h-24 mb-6 bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30 rounded-2xl flex items-center justify-center"
                      >
                        <ChartLine className="w-12 h-12 text-indigo-600 dark:text-indigo-400" />
                      </motion.div>
                      <h4 className="text-xl font-bold text-gray-800 dark:text-gray-200 mb-2">
                        Ready to Predict
                      </h4>
                      <p className="text-gray-600 dark:text-gray-400 max-w-sm">
                        Enter drug SMILES and protein sequence, then click "Predict Interaction" to see results
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
}

// Enhanced Results Display Component
function ResultsDisplay({ results }) {
  const { probability, binding, confidence } = results;
  const pActivity = -Math.log10(probability * 1e-6);
  const ic50 = Math.pow(10, -pActivity) * 1e9;

  let interpretation = '';
  let interpretationColor = '';
  let interpretationIcon = null;
  
  if (probability > 0.7) {
    interpretation = 'Strong Binding Predicted: High likelihood of therapeutic interaction. Recommended for further experimental validation and in vitro testing.';
    interpretationColor = 'from-green-500 to-emerald-500';
    interpretationIcon = CheckCircle2;
  } else if (probability > 0.5) {
    interpretation = 'Moderate Binding Predicted: Shows promising interaction potential. Consider molecular optimization and structure-activity relationship studies.';
    interpretationColor = 'from-yellow-500 to-orange-500';
    interpretationIcon = AlertTriangle;
  } else {
    interpretation = 'Weak Binding Predicted: Low probability of meaningful interaction. This drug-target pair may require significant modification or alternative targets.';
    interpretationColor = 'from-red-500 to-pink-500';
    interpretationIcon = XCircle;
  }

  const InterpretationIcon = interpretationIcon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
            <ChartLine className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
            Prediction Results
          </h3>
        </div>
    
      </div>

      {/* Main Probability Card */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 200 }}
        className={`relative overflow-hidden bg-gradient-to-br ${interpretationColor} rounded-2xl p-8 text-white shadow-xl`}
      >
        <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-16 translate-x-16" />
        <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/10 rounded-full translate-y-12 -translate-x-12" />
        
        <div className="relative text-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="text-6xl md:text-7xl font-extrabold mb-2"
          >
            {(probability * 100).toFixed(1)}%
          </motion.div>
          <p className="text-xl font-semibold opacity-90">Binding Probability</p>
          <p className="text-sm opacity-75 mt-2">
            Likelihood of drug-protein interaction
          </p>
        </div>
      </motion.div>

      {/* Status Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Binding Status */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 rounded-xl p-6 border border-gray-200 dark:border-gray-600"
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Interaction Status
            </span>
            {binding === 'Yes' ? (
              <CheckCircle2 className="w-5 h-5 text-green-500" />
            ) : (
              <XCircle className="w-5 h-5 text-red-500" />
            )}
          </div>
          <div
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-full font-bold text-lg ${
              binding === 'Yes'
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
            }`}
          >
            <Circle className="w-3 h-3 fill-current" />
            {binding === 'Yes' ? 'Binding' : 'Non-Binding'}
          </div>
        </motion.div>

        {/* Confidence Score */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 rounded-xl p-6 border border-gray-200 dark:border-gray-600"
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Model Confidence
            </span>
            <Brain className="w-5 h-5 text-purple-500" />
          </div>
          <div className="relative h-3 bg-gray-300 dark:bg-gray-600 rounded-full overflow-hidden mb-3">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidence * 100}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
              className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
            />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {(confidence * 100).toFixed(1)}%
          </div>
        </motion.div>
      </div>

      {/* Detailed Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-indigo-200 dark:border-indigo-800"
      >
        <h4 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2 mb-4">
          <Microscope className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          Detailed Metrics
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Raw Score" value={probability.toFixed(4)} icon="📊" />
          <MetricCard label="pActivity" value={pActivity.toFixed(2)} icon="🧪" />
          <MetricCard label="Est. IC50" value={`${ic50.toFixed(1)} nM`} icon="💊" />
          <MetricCard 
            label="Suitability" 
            value={binding === 'Yes' ? 'High' : 'Low'} 
            icon={binding === 'Yes' ? '✅' : '❌'}
          />
        </div>
      </motion.div>

      {/* Interpretation Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className={`relative overflow-hidden bg-gradient-to-br ${interpretationColor} rounded-xl p-6 text-white shadow-lg`}
      >
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-12 h-12 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center">
            <InterpretationIcon className="w-6 h-6" />
          </div>
          <div className="flex-1">
            <h4 className="text-lg font-bold mb-2 flex items-center gap-2">
              Analysis Summary
            </h4>
            <p className="text-sm leading-relaxed opacity-95">
              {interpretation}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

// Metric Card Component
function MetricCard({ label, value, icon }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05, y: -2 }}
      className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md border border-gray-200 dark:border-gray-700 text-center"
    >
      <div className="text-2xl mb-2">{icon}</div>
      <div className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
        {label}
      </div>
      <div className="text-lg font-bold text-gray-900 dark:text-white">
        {value}
      </div>
    </motion.div>
  );
}
