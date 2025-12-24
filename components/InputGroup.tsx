import React from 'react';

interface InputGroupProps {
  id: string;
  label: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  placeholder?: string;
  height?: string;
  icon?: React.ReactNode;
}

export const InputGroup: React.FC<InputGroupProps> = ({ 
  id, 
  label, 
  value, 
  onChange, 
  placeholder, 
  height = "h-32",
  icon
}) => {
  return (
    <div className="flex flex-col gap-2 group">
      <label htmlFor={id} className="flex items-center gap-2 text-sm font-medium text-slate-400 group-focus-within:text-indigo-400 transition-colors">
        {icon && <span className="text-slate-500 group-focus-within:text-indigo-500 transition-colors">{icon}</span>}
        {label}
      </label>
      <textarea
        id={id}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className={`w-full ${height} px-4 py-3 bg-slate-950/50 border border-slate-700 rounded-lg 
        focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none 
        text-slate-200 placeholder-slate-600 resize-none transition-all duration-200
        hover:border-slate-600`}
      />
    </div>
  );
};
