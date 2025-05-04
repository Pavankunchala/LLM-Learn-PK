// components/ClassSelector.tsx
// (No changes needed, assumes AVAILABLE_CLASSES keys match backend strings)

import React from 'react';
import { motion } from 'framer-motion';
import { Check } from 'lucide-react';
import { useDetectionStore } from '../store/detectionStore';

interface ClassSelectorProps {
  classes: Record<string, string>; // e.g., { "person": "Person", ... }
}

const ClassSelector: React.FC<ClassSelectorProps> = ({ classes }) => {
  const { selectedClasses, toggleClass } = useDetectionStore();

  return (
    <div className="max-h-48 overflow-y-auto pr-1 space-y-1 custom-scrollbar"> {/* Added custom-scrollbar class */}
      {Object.entries(classes).map(([id, name]) => ( // 'id' here is the string key like "person"
        <motion.div
          key={id}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          className={`flex items-center p-2 rounded-md cursor-pointer transition-colors ${
            selectedClasses.includes(id)
              ? 'bg-primary-900 hover:bg-primary-800'
              : 'hover:bg-gray-700'
          }`}
          onClick={() => toggleClass(id)} // Toggle using the string ID
        >
          <div
            className={`w-4 h-4 rounded flex items-center justify-center mr-2 ${
              selectedClasses.includes(id)
                ? 'bg-primary-500 text-white' // Checkmark is white
                : 'border border-gray-500'
            }`}
          >
            {selectedClasses.includes(id) && <Check size={12} />}
          </div>
          <span className="text-sm">{name}</span>
        </motion.div>
      ))}
    </div>
  );
};

export default ClassSelector;