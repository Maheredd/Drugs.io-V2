import { useState } from "react";
import DrawMolecule from "./DrawMolecule";
import { motion } from "framer-motion";
import Lottie from "lottie-react";
import predictAnimation from "../assets/Predict.json";
import loadingAnimation from "../assets/Loading.json";
import PredictionPanel from "./PredictionPanel";
import DrugTarget from "./DrugTarget";
import DrugCombination from "./DrugCombination";
import Predict from "./Predict";



const MainPredict = () => {
  const [selectedOption, setSelectedOption] = useState("text");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  return (
    <>
      {/* Fullscreen loading overlay */}
      {loading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-white dark:bg-gray-900 bg-opacity-80 backdrop-blur-sm">
          <Lottie animationData={loadingAnimation} loop className="w-72 h-72 sm:w-96 sm:h-96" />
        </div>
      )}

      <motion.div
        className={`p-4 sm:p-6 max-w-6xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-2xl transition-all duration-300 ${loading ? "pointer-events-none opacity-40" : ""
          }`}
      >
        <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center text-gray-900 text-blue-400">
          Drug Discovery
        </h1>

        <div className="">
          <div className="w-full ">
            {/* Existing decorative animation */}
            <Lottie animationData={predictAnimation} loop className="w-full h-64 sm:h-80 md:h-[400px]" />
          </div>

          <div className="w-full ">
            <div className="flex flex-wrap gap-2 justify-center md:justify-center mb-4">
              {["ADMET", "Drug-Target", "Drug-Combination"].map((option) => (
                <button
                  key={option}
                  className={`px-4 py-2 rounded font-medium ${selectedOption === option ? "bg-blue-600 text-white" : "bg-blue-100 text-blue-800"
                    }`}
                  onClick={() => {
                    setSelectedOption(option);
                  }}
                >
                  {option === "ADMET"
                      ? "ADMET"
                      : option === "Drug-Target"
                      ? "Drug Target"
                        : "Drug Combination"  
                        }
                </button>
              ))}
            </div>

            {/* Input Sections */}
            
             {selectedOption === "Drug-Target" && (
             <DrugTarget/>
            )}

            {selectedOption === "Drug-Combination" && (
             <DrugCombination/>
            )}

            {selectedOption === "ADMET" && (
             <Predict/>
            )}

            {error && <p className="mt-2 text-red-600 font-medium">{error}</p>}
          </div>
        </div>

      </motion.div>
    </>
  );
};

export default MainPredict;