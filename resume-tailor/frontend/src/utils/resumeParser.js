/**
 * Enhanced resume parsing utility to better preserve structure and content
 */

// Regex patterns for identifying sections in LaTeX resumes
const SECTION_PATTERNS = {
    experience: /\\section\*{Experience}(.*?)(?=\\section\*{|\\end{document}|$)/s,
    education: /\\section\*{Education}(.*?)(?=\\section\*{|\\end{document}|$)/s,
    skills: /\\section\*{(Technical\s+Skills|Skills)}(.*?)(?=\\section\*{|\\end{document}|$)/s,
    projects: /\\section\*{Projects}(.*?)(?=\\section\*{|\\end{document}|$)/s,
    links: /\\href{([^}]+)}{([^}]+)}/g,
    companies: /\\item\\textbf{([^}]+)}.*?\\hfill\s*([^\\]+)/g
  };
  
  /**
   * Parse a LaTeX resume to extract important sections and formatting
   * @param {string} latexContent - The LaTeX resume content
   * @returns {Object} Parsed resume sections and metadata
   */
  export const parseLatexResume = (latexContent) => {
    if (!latexContent) return null;
  
    const sections = {};
    const metadata = {
      hasLinks: false,
      sectionCount: 0,
      linkCount: 0,
      companyCount: 0
    };
  
    // Extract each section
    for (const [sectionName, pattern] of Object.entries(SECTION_PATTERNS)) {
      if (sectionName === 'links' || sectionName === 'companies') continue;
      
      const match = latexContent.match(pattern);
      if (match) {
        sections[sectionName] = match[1] || match[2] || '';
        metadata.sectionCount++;
      }
    }
    
    // Extract links
    const links = [];
    let linkMatch;
    while ((linkMatch = SECTION_PATTERNS.links.exec(latexContent)) !== null) {
      links.push({
        url: linkMatch[1],
        text: linkMatch[2]
      });
      metadata.linkCount++;
    }
    
    if (links.length > 0) {
      sections.links = links;
      metadata.hasLinks = true;
    }
    
    // Extract companies from experience section
    const companies = [];
    if (sections.experience) {
      let companyMatch;
      while ((companyMatch = SECTION_PATTERNS.companies.exec(sections.experience)) !== null) {
        companies.push({
          name: companyMatch[1],
          date: companyMatch[2].trim()
        });
        metadata.companyCount++;
      }
      
      if (companies.length > 0) {
        sections.companies = companies;
      }
    }
    
    return { sections, metadata };
  };
  
  /**
   * Check if a tailored resume preserves all original sections and structure
   * @param {string} originalLatex - The original LaTeX resume
   * @param {string} tailoredLatex - The tailored LaTeX resume
   * @returns {Object} Validation results and issues found
   */
  export const validateResumePreservation = (originalLatex, tailoredLatex) => {
    const original = parseLatexResume(originalLatex);
    const tailored = parseLatexResume(tailoredLatex);
    
    if (!original || !tailored) return { valid: false, issues: ['Invalid LaTeX content'] };
    
    const issues = [];
    
    // Check if all sections are preserved
    for (const section in original.sections) {
      if (section === 'links' || section === 'companies') continue;
      
      if (!tailored.sections[section]) {
        issues.push(`Missing section: ${section}`);
      }
    }
    
    // Check if all companies are preserved
    if (original.sections.companies && tailored.sections.companies) {
      const originalCompanies = original.sections.companies.map(c => c.name);
      const tailoredCompanies = tailored.sections.companies.map(c => c.name);
      
      const missingCompanies = originalCompanies.filter(c => !tailoredCompanies.includes(c));
      if (missingCompanies.length > 0) {
        issues.push(`Missing companies: ${missingCompanies.join(', ')}`);
      }
      
      // Check for duplicate companies
      const companyCounts = {};
      tailoredCompanies.forEach(company => {
        companyCounts[company] = (companyCounts[company] || 0) + 1;
      });
      
      const duplicates = Object.entries(companyCounts)
        .filter(([_, count]) => count > 1)
        .map(([company, count]) => `${company} (${count} times)`);
      
      if (duplicates.length > 0) {
        issues.push(`Duplicate companies: ${duplicates.join(', ')}`);
      }
    }
    
    // Check if links are preserved
    if (original.metadata.hasLinks && !tailored.metadata.hasLinks) {
      issues.push('Links were lost in the tailored resume');
    } else if (original.metadata.linkCount > tailored.metadata.linkCount) {
      issues.push(`Missing links: ${original.metadata.linkCount - tailored.metadata.linkCount} links were lost`);
    }
    
    return {
      valid: issues.length === 0,
      issues,
      originalMetadata: original.metadata,
      tailoredMetadata: tailored.metadata
    };
  };
  
  /**
   * Extract all links from a LaTeX document
   * @param {string} latexContent - The LaTeX content
   * @returns {Array} Array of link objects with url and text
   */
  export const extractLinks = (latexContent) => {
    const links = [];
    const linkPattern = /\\href{([^}]+)}{([^}]+)}/g;
    
    let match;
    while ((match = linkPattern.exec(latexContent)) !== null) {
      links.push({
        url: match[1],
        text: match[2]
      });
    }
    
    return links;
  };
  
  /**
   * Fix common issues in tailored resumes by preserving original structure
   * @param {string} originalLatex - The original LaTeX resume
   * @param {string} tailoredLatex - The tailored LaTeX resume
   * @returns {string} Fixed LaTeX resume content
   */
  export const fixResumeIssues = (originalLatex, tailoredLatex) => {
    if (!originalLatex || !tailoredLatex) return tailoredLatex;
    
    let fixedLatex = tailoredLatex;
    const validation = validateResumePreservation(originalLatex, tailoredLatex);
    
    // If no issues, return the original tailored resume
    if (validation.valid) return tailoredLatex;
    
    const original = parseLatexResume(originalLatex);
    const tailored = parseLatexResume(tailoredLatex);
    
    // Fix missing sections by copying them from the original
    for (const section in original.sections) {
      if (section === 'links' || section === 'companies') continue;
      
      if (!tailored.sections[section]) {
        const sectionName = section.charAt(0).toUpperCase() + section.slice(1);
        const sectionRegex = new RegExp(`\\\\section\\*{${sectionName}}.*?(?=\\\\section\\*{|\\\\end{document}|$)`, 's');
        
        // Find a good position to insert the section
        let insertPosition = fixedLatex.lastIndexOf('\\section*{');
        if (insertPosition === -1) {
          insertPosition = fixedLatex.indexOf('\\end{document}');
        } else {
          // Find the position of the next section after this one in the original
          const originalSections = Object.keys(original.sections)
            .filter(s => s !== 'links' && s !== 'companies');
          const sectionIndex = originalSections.indexOf(section);
          
          if (sectionIndex < originalSections.length - 1) {
            const nextSection = originalSections[sectionIndex + 1];
            const nextSectionName = nextSection.charAt(0).toUpperCase() + nextSection.slice(1);
            const nextSectionMatch = fixedLatex.match(new RegExp(`\\\\section\\*{${nextSectionName}}`));
            
            if (nextSectionMatch) {
              insertPosition = nextSectionMatch.index;
            }
          }
        }
        
        // Extract the section content from the original
        const sectionMatch = originalLatex.match(sectionRegex);
        if (sectionMatch && insertPosition !== -1) {
          const sectionContent = sectionMatch[0];
          fixedLatex = fixedLatex.slice(0, insertPosition) + 
                       sectionContent + '\n\n' + 
                       fixedLatex.slice(insertPosition);
        }
      }
    }
    
    // Fix missing links by restoring them from the original
    if (original.sections.links && (!tailored.sections.links || tailored.metadata.linkCount < original.metadata.linkCount)) {
      // Get all links from the original
      const originalLinks = original.sections.links;
      
      // Check each link to see if it exists in the tailored resume
      originalLinks.forEach(link => {
        const linkText = `\\href{${link.url}}{${link.text}}`;
        const plainText = link.text;
        
        // If the plain text exists but not as a hyperlink, replace it with the hyperlink
        if (!fixedLatex.includes(linkText) && fixedLatex.includes(plainText)) {
          fixedLatex = fixedLatex.replace(
            new RegExp(plainText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), 
            linkText
          );
        }
      });
    }
    
    // Fix duplicate companies in the experience section
    if (tailored.sections.companies) {
      const companyCounts = {};
      tailored.sections.companies.forEach(company => {
        companyCounts[company.name] = (companyCounts[company.name] || 0) + 1;
      });
      
      const duplicates = Object.entries(companyCounts).filter(([_, count]) => count > 1);
      
      if (duplicates.length > 0) {
        // Extract the experience section
        const expMatch = fixedLatex.match(SECTION_PATTERNS.experience);
        if (expMatch) {
          const originalExpMatch = originalLatex.match(SECTION_PATTERNS.experience);
          if (originalExpMatch) {
            // Replace the experience section with the original one
            fixedLatex = fixedLatex.replace(
              expMatch[0],
              originalExpMatch[0]
            );
          }
        }
      }
    }
    
    return fixedLatex;
  };