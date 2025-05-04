import React from 'react';
import { Camera, Monitor, Github } from 'lucide-react';
import { motion } from 'framer-motion';

const AppHeader = () => {
  return (
    <motion.header 
      className="bg-gray-800 shadow-md"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <motion.div 
            whileHover={{ rotate: 10 }}
            className="p-1.5 bg-primary-500 rounded-lg"
          >
            <Camera size={24} className="text-white" />
          </motion.div>
          <div>
            <h1 className="text-xl font-bold text-white">DetectTrack</h1>
            <p className="text-xs text-gray-400">Object Detection & Tracking</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noopener noreferrer"
            className="flex items-center text-gray-400 hover:text-white transition-colors"
          >
            <Github size={20} className="mr-1" />
            <span className="text-sm hidden sm:inline">GitHub</span>
          </a>
          
          <div className="flex items-center text-sm text-gray-400">
            <Monitor size={16} className="mr-1" />
            <span className="hidden sm:inline">Live Demo</span>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default AppHeader;