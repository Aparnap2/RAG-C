"""
Document normalizer for standardizing data from various sources.
Handles ACL mapping, PII scrubbing, and document preparation.
"""
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Pattern

logger = logging.getLogger(__name__)

class PIIDetector:
    """
    Detects and scrubs personally identifiable information (PII) from text.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, Pattern]:
        """Compile regex patterns for PII detection"""
        patterns = {}
        
        # Email addresses
        patterns["email"] = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone numbers (various formats)
        patterns["phone"] = re.compile(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
        
        # Social Security Numbers
        patterns["ssn"] = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')
        
        # Credit card numbers
        patterns["credit_card"] = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
        
        # IP addresses
        patterns["ip"] = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        
        # Add custom patterns from config
        for name, pattern in self.config.get("custom_patterns", {}).items():
            try:
                patterns[name] = re.compile(pattern)
            except re.error:
                logger.error(f"Invalid regex pattern for {name}: {pattern}")
                
        return patterns
        
    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text
        
        Args:
            text: Text to scan
            
        Returns:
            Dictionary of PII type to list of matches
        """
        results = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[pii_type] = matches
                
        return results
        
    def scrub(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Scrub PII from text
        
        Args:
            text: Text to scrub
            replacement: Replacement string for PII
            
        Returns:
            Scrubbed text
        """
        scrubbed = text
        
        for pii_type, pattern in self.patterns.items():
            scrubbed = pattern.sub(replacement, scrubbed)
            
        return scrubbed


class ACLMapper:
    """
    Maps source-specific ACLs to canonical ACL format.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mappings = config.get("acl_mappings", {})
        
    def map_acls(self, source_acls: List[str], source_tool: str, tenant_id: str) -> List[str]:
        """
        Map source-specific ACLs to canonical ACLs
        
        Args:
            source_acls: Source ACLs
            source_tool: Source tool identifier
            tenant_id: Tenant identifier
            
        Returns:
            Canonical ACLs
        """
        # Start with tenant-level ACL
        canonical_acls = [f"tenant:{tenant_id}"]
        
        # Get tool-specific mappings
        tool_mappings = self.mappings.get(source_tool, {})
        
        # Map each source ACL
        for acl in source_acls:
            # Check for direct mapping
            if acl in tool_mappings:
                canonical_acls.append(tool_mappings[acl])
            else:
                # Check for pattern mappings
                for pattern, mapping in tool_mappings.get("patterns", {}).items():
                    if re.match(pattern, acl):
                        # Replace capture groups if present
                        if re.search(r'\$\d+', mapping):
                            match = re.match(pattern, acl)
                            if match:
                                for i, group in enumerate(match.groups(), 1):
                                    mapping = mapping.replace(f"${i}", group)
                        canonical_acls.append(mapping)
                        break
                else:
                    # No mapping found, use the original with a prefix
                    canonical_acls.append(f"{source_tool}:{acl}")
                    
        return canonical_acls


class Normalizer:
    """
    Normalizes documents from various sources into a canonical format.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pii_detector = PIIDetector(config.get("pii", {}))
        self.acl_mapper = ACLMapper(config)
        self.schema_version = config.get("schema_version", "1.0")
        
    async def normalize(self, document: Dict[str, Any], scrub_pii: bool = True) -> Dict[str, Any]:
        """
        Normalize a document
        
        Args:
            document: Source document
            scrub_pii: Whether to scrub PII
            
        Returns:
            Normalized document
        """
        # Extract required fields
        tenant_id = document.get("tenant_id")
        source_tool = document.get("source_tool")
        source_id = document.get("source_id")
        
        if not all([tenant_id, source_tool, source_id]):
            raise ValueError("Document missing required fields: tenant_id, source_tool, source_id")
            
        # Extract content
        content = document.get("content", "")
        
        # Scrub PII if requested
        if scrub_pii and content:
            content = self.pii_detector.scrub(content)
            
        # Map ACLs
        source_acls = document.get("acl", [])
        canonical_acls = self.acl_mapper.map_acls(source_acls, source_tool, tenant_id)
        
        # Extract metadata
        metadata = document.get("metadata", {})
        
        # Extract timestamps
        ts_source = document.get("ts_source") or document.get("timestamp") or datetime.now().isoformat()
        ts_ingested = document.get("ts_ingested") or datetime.now().isoformat()
        
        # Compute checksum if not provided
        checksum = document.get("checksum")
        if not checksum:
            checksum_doc = {
                "source_id": source_id,
                "content": content,
                "metadata": metadata,
                "ts_source": ts_source
            }
            checksum = hashlib.md5(json.dumps(checksum_doc, sort_keys=True).encode()).hexdigest()
            
        # Create normalized document
        normalized = {
            "id": f"{tenant_id}:{source_tool}:{source_id}",
            "tenant_id": tenant_id,
            "source_tool": source_tool,
            "source_id": source_id,
            "content": content,
            "metadata": metadata,
            "acl": canonical_acls,
            "ts_source": ts_source,
            "ts_ingested": ts_ingested,
            "checksum": checksum,
            "schema_version": self.schema_version
        }
        
        return normalized
        
    async def process_batch(self, documents: List[Dict[str, Any]], scrub_pii: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of documents
        
        Args:
            documents: List of source documents
            scrub_pii: Whether to scrub PII
            
        Returns:
            List of normalized documents
        """
        normalized = []
        
        for document in documents:
            try:
                normalized_doc = await self.normalize(document, scrub_pii)
                normalized.append(normalized_doc)
            except Exception as e:
                logger.error(f"Error normalizing document: {str(e)}")
                # Skip this document
                
        return normalized