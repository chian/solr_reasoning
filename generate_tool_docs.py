#!/usr/bin/env python3
"""
Tool Documentation Generator for BV-BRC MCP Servers

This script extracts all tool information from the MCP server files and generates
comprehensive, accurate documentation for use in the LLM prompt.
"""

import ast
import inspect
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

class ToolExtractor:
    def __init__(self):
        self.servers = {
            "DATA_RETRIEVAL": "P3_TOOLS_DATA_RETRIEVAL.py",
            "COMPUTATIONAL": "P3_TOOLS_COMPUTATIONAL.py", 
            "UTILITIES": "P3_TOOLS_UTILITIES.py",
            "REST_API": "BVBRC_API.py"
        }
        self.extracted_tools = {}
    
    def extract_all_tools(self):
        """Extract tools from all MCP servers"""
        for server_name, server_file in self.servers.items():
            print(f"Extracting tools from {server_file}...")
            tools = self.extract_tools_from_file(server_file)
            self.extracted_tools[server_name] = tools
            print(f"Found {len(tools)} tools in {server_name}")
        
        return self.extracted_tools
    
    def extract_tools_from_file(self, filename: str) -> List[Dict[str, Any]]:
        """Extract MCP tools from a Python file"""
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            tools = []
            
            # Find all functions decorated with @mcp.tool() (both sync and async)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if function has @mcp.tool() decorator
                    if self.has_mcp_tool_decorator(node):
                        tool_info = self.extract_tool_info(node, content)
                        if tool_info:
                            tools.append(tool_info)
            
            return tools
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return []
    
    def has_mcp_tool_decorator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if function has @mcp.tool() decorator"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if (isinstance(decorator.func.value, ast.Name) and 
                        decorator.func.value.id == 'mcp' and 
                        decorator.func.attr == 'tool'):
                        return True
            elif isinstance(decorator, ast.Attribute):
                if (isinstance(decorator.value, ast.Name) and 
                    decorator.value.id == 'mcp' and 
                    decorator.attr == 'tool'):
                    return True
        return False
    
    def extract_tool_info(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], content: str) -> Dict[str, Any]:
        """Extract detailed information about a tool function"""
        try:
            # Get function name
            func_name = node.name
            
            # Get docstring
            docstring = ast.get_docstring(node) or ""
            
            # Extract parameters with types and defaults
            params = self.extract_parameters(node)
            
            # Extract return type
            return_type = self.extract_return_type(node)
            
            # Parse docstring for additional info
            doc_info = self.parse_docstring(docstring)
            
            return {
                'name': func_name,
                'parameters': params,
                'return_type': return_type,
                'docstring': docstring,
                'description': doc_info.get('description', ''),
                'args_doc': doc_info.get('args', {}),
                'examples': doc_info.get('examples', [])
            }
            
        except Exception as e:
            print(f"Error extracting info for {node.name}: {e}")
            return None
    
    def extract_parameters(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract parameter information from function definition"""
        params = []
        args = node.args
        
        # Get default values
        defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults
        
        for i, arg in enumerate(args.args):
            # Skip 'self' parameter
            if arg.arg == 'self':
                continue
                
            param_info = {
                'name': arg.arg,
                'type': self.extract_type_annotation(arg.annotation),
                'default': self.extract_default_value(defaults[i]),
                'required': defaults[i] is None
            }
            params.append(param_info)
        
        return params
    
    def extract_type_annotation(self, annotation) -> str:
        """Extract type annotation as string"""
        if annotation is None:
            return "Any"
        
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            elif isinstance(annotation, ast.Subscript):
                # Handle List[str], Optional[str], etc.
                if isinstance(annotation.value, ast.Name):
                    base_type = annotation.value.id
                    if isinstance(annotation.slice, ast.Name):
                        inner_type = annotation.slice.id
                        return f"{base_type}[{inner_type}]"
                    elif isinstance(annotation.slice, ast.Constant):
                        inner_type = annotation.slice.value
                        return f"{base_type}[{inner_type}]"
                return "Complex"
            else:
                return "Unknown"
        except:
            return "Any"
    
    def extract_default_value(self, default_node) -> Any:
        """Extract default value from AST node"""
        if default_node is None:
            return None
        
        try:
            if isinstance(default_node, ast.Constant):
                return default_node.value
            elif isinstance(default_node, ast.Name):
                if default_node.id == 'None':
                    return None
                elif default_node.id == 'True':
                    return True
                elif default_node.id == 'False':
                    return False
                else:
                    return default_node.id
            elif isinstance(default_node, ast.List):
                return []
            elif isinstance(default_node, ast.Dict):
                return {}
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def extract_return_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Extract return type annotation"""
        if node.returns:
            return self.extract_type_annotation(node.returns)
        return "Any"
    
    def parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse docstring to extract structured information"""
        if not docstring:
            return {}
        
        lines = docstring.strip().split('\n')
        result = {
            'description': '',
            'args': {},
            'examples': []
        }
        
        current_section = 'description'
        description_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.lower().startswith('args:') or line.lower().startswith('arguments:'):
                current_section = 'args'
                result['description'] = '\n'.join(description_lines).strip()
            elif line.lower().startswith('returns:') or line.lower().startswith('return:'):
                current_section = 'returns'
            elif line.lower().startswith('examples:') or line.lower().startswith('example:'):
                current_section = 'examples'
            elif current_section == 'description':
                description_lines.append(line)
            elif current_section == 'args' and ':' in line:
                # Parse argument documentation
                parts = line.split(':', 1)
                if len(parts) == 2:
                    arg_name = parts[0].strip()
                    arg_desc = parts[1].strip()
                    result['args'][arg_name] = arg_desc
        
        if not result['description']:
            result['description'] = '\n'.join(description_lines).strip()
        
        return result

class DocumentationGenerator:
    def __init__(self, extracted_tools: Dict[str, List[Dict[str, Any]]]):
        self.tools = extracted_tools
    
    def generate_complete_documentation(self) -> str:
        """Generate complete tool documentation for the LLM prompt"""
        doc_parts = []
        
        # Header
        doc_parts.append("AVAILABLE TOOLS (tool_name) AND THEIR PARAMETERS:")
        doc_parts.append("")
        
        # Generate documentation for each server
        server_sections = {
            "DATA_RETRIEVAL": "DATA RETRIEVAL TOOLS",
            "COMPUTATIONAL": "COMPUTATIONAL TOOLS", 
            "UTILITIES": "UTILITY TOOLS",
            "REST_API": "REST API TOOLS"
        }
        
        tool_counter = 1
        
        for server_name, section_title in server_sections.items():
            if server_name in self.tools and self.tools[server_name]:
                doc_parts.append(f"{section_title}:")
                
                for tool in self.tools[server_name]:
                    tool_doc = self.generate_tool_documentation(tool, tool_counter)
                    doc_parts.append(tool_doc)
                    doc_parts.append("")
                    tool_counter += 1
        
        return "\n".join(doc_parts)
    
    def generate_tool_documentation(self, tool: Dict[str, Any], counter: int) -> str:
        """Generate documentation for a single tool"""
        lines = []
        
        # Tool name and number
        lines.append(f"{counter}. {tool['name']}")
        
        # Parameters
        if tool['parameters']:
            param_strs = []
            for param in tool['parameters']:
                param_str = self.format_parameter(param)
                param_strs.append(param_str)
            
            lines.append(f"   - Parameters: {', '.join(param_strs)}")
        else:
            lines.append("   - Parameters: {} (no parameters)")
        
        # Description (from docstring)
        if tool['description']:
            # Clean up description - take first sentence or first line
            desc = tool['description'].split('\n')[0].strip()
            if len(desc) > 100:
                desc = desc[:97] + "..."
            lines.append(f"   - Description: {desc}")
        
        # Example
        example = self.generate_example(tool)
        if example:
            lines.append(f"   - Example: {example}")
        
        return "\n".join(lines)
    
    def format_parameter(self, param: Dict[str, Any]) -> str:
        """Format a parameter for documentation"""
        name = param['name']
        param_type = param['type']
        default = param['default']
        required = param['required']
        
        # Clean up type names
        type_map = {
            'Optional[str]': 'str, optional',
            'Optional[int]': 'int, optional', 
            'Optional[bool]': 'bool, optional',
            'Optional[List[str]]': 'list, optional',
            'List[str]': 'list',
            'List[int]': 'list',
            'Dict[str, str]': 'dict',
            'Dict[str, Any]': 'dict'
        }
        
        display_type = type_map.get(param_type, param_type.lower())
        
        if not required and default is not None:
            if isinstance(default, str):
                return f"{name} ({display_type}, default='{default}')"
            else:
                return f"{name} ({display_type}, default={default})"
        elif not required:
            return f"{name} ({display_type}, optional)"
        else:
            return f"{name} ({display_type})"
    
    def generate_example(self, tool: Dict[str, Any]) -> str:
        """Generate a usage example for the tool"""
        if not tool['parameters']:
            return "{}"
        
        example_params = {}
        
        for param in tool['parameters']:
            name = param['name']
            param_type = param['type']
            default = param['default']
            
            # Generate example values based on parameter name and type
            if 'species' in name.lower():
                example_params[name] = "Escherichia coli"
            elif 'genome_id' in name.lower():
                if 'list' in param_type.lower():
                    example_params[name] = ["83333.111"]
                else:
                    example_params[name] = "83333.111"
            elif 'feature_id' in name.lower():
                if 'list' in param_type.lower():
                    example_params[name] = ["fig|83333.111.peg.1"]
                else:
                    example_params[name] = "fig|83333.111.peg.1"
            elif 'limit' in name.lower():
                example_params[name] = 10
            elif 'sequence' in name.lower() and 'type' not in name.lower():
                example_params[name] = "MKTVRQERLK..."
            elif param['required'] and not default:
                # Add required parameters with generic examples
                if 'str' in param_type.lower():
                    example_params[name] = "example_value"
                elif 'int' in param_type.lower():
                    example_params[name] = 100
                elif 'bool' in param_type.lower():
                    example_params[name] = True
                elif 'list' in param_type.lower():
                    example_params[name] = ["example"]
        
        if example_params:
            import json
            return json.dumps(example_params)
        else:
            return "{}"

def main():
    """Main function to generate tool documentation"""
    print("BV-BRC MCP Tool Documentation Generator")
    print("=" * 50)
    
    # Extract tools from all servers
    extractor = ToolExtractor()
    extracted_tools = extractor.extract_all_tools()
    
    # Generate documentation
    generator = DocumentationGenerator(extracted_tools)
    documentation = generator.generate_complete_documentation()
    
    # Save to file
    output_file = "complete_tool_reference.txt"
    with open(output_file, 'w') as f:
        f.write(documentation)
    
    print(f"\nComplete tool documentation saved to: {output_file}")
    
    # Print summary
    total_tools = sum(len(tools) for tools in extracted_tools.values())
    print(f"\nSummary:")
    print(f"Total tools extracted: {total_tools}")
    for server_name, tools in extracted_tools.items():
        print(f"  {server_name}: {len(tools)} tools")
    
    print(f"\nDocumentation length: {len(documentation)} characters")
    print("\nYou can now update your prompt to use this complete tool reference!")

if __name__ == "__main__":
    main() 