import { FaReact, FaGithub } from "react-icons/fa";
import { SiTailwindcss, SiFlask } from "react-icons/si";
import { MdScience } from "react-icons/md";
import { TbDatabaseCog } from "react-icons/tb";
import { motion } from "framer-motion";
import React, { useState, useEffect } from "react";

const Footer = () => {
  const [showCite, setShowCite] = useState(false);
  const [accessedDate, setAccessedDate] = useState("");

  useEffect(() => {
    const date = new Date();
    const month = date.toLocaleDateString("default", { month: "long" });
    const year = date.getFullYear();
    setAccessedDate(`${month} ${year}`);
  }, []);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert("Copied to Clipboard!");
  };

  const generalCitation = `HARSHITHA M G, et al. (2025). ADMET-X: AI-Driven Platform for In-Silico ADMET Prediction. [Online]. Available: https://admet-x.vercel.app`;

  const ieeeCitation = `[1] HARSHITHA M G, et al., “ADMET-X: AI-Driven Platform for In-Silico ADMET Prediction,” 2025. [Online]. Available: https://admet-x.vercel.app`;

  const bibtexCitation = `@software{admetx2025,
  author = {HARSHITHA M G},
  title = {ADMET-X: AI-Driven Platform for In-Silico ADMET Prediction},
  year = {2025},
  url = {https://admet-x.vercel.app},
  note = {Accessed: ${accessedDate}}
}`;

  return (
    <>
      <motion.footer
        className="bg-blue-100 dark:bg-gray-800 text-gray-800 dark:text-white py-4 shadow-md border-t border-gray-300 dark:border-gray-700"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-6 sm:gap-4">

          {/* ABOUT SECTION */}
          <div className="flex-1 text-center sm:text-left space-y-2 pb-4 border-b border-gray-300 dark:border-gray-700 sm:border-none">
            <h3 className="font-semibold text-lg tracking-wide">ABOUT</h3>
            <p className="text-sm dark:text-gray-300 leading-relaxed">
              <a
                href="https://admet-x.vercel.app"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 dark:text-blue-400 font-semibold hover:underline"
              >
                DRUG.IO
              </a>{" "}
             is an AI-powered drug discovery platform that predicts ADMET properties, drug combinations, and drug–target interactions to accelerate therapeutic development.
            </p>
            
          </div>

          {/* DEV TOOLS */}
          <div className="flex-1 text-center sm:text-left pb-4 border-b border-gray-300 dark:border-gray-700 sm:border-none">
            <h3 className="font-semibold text-lg mb-2 tracking-wide">DEV TOOLS</h3>
            <div className="flex justify-center sm:justify-start gap-4 text-4xl">
              <FaReact className="text-cyan-500 dark:text-cyan-400 hover:scale-110 transition-transform" title="React" />
              <SiTailwindcss className="text-sky-400 dark:text-sky-400 hover:scale-110 transition-transform" title="Tailwind CSS" />
              <SiFlask className="text-black dark:text-white hover:scale-110 transition-transform" title="Flask" />
              <MdScience className="text-green-600 dark:text-green-400 hover:scale-110 transition-transform" title="RDKit" />
              <TbDatabaseCog className="text-purple-600 dark:text-purple-400 hover:scale-110 transition-transform" title="TDC" />
            </div>
          </div>

          
        </div>

        <div className="mt-4 text-center text-sm font-bold dark:text-gray-400">
          © {new Date().getFullYear()} DRUG.IO — All rights reserved.
        </div>
      </motion.footer>

      
      
    </>
  );
};

export default Footer;
