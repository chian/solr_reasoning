#!/bin/bash
# Update P3-Tools from GitHub Repository
# Downloads the latest versions of key p3-tools scripts

echo "🔄 Updating P3-Tools from GitHub repository..."

# Base URL for the scripts
BASE_URL="https://raw.githubusercontent.com/BV-BRC/p3_cli/refs/heads/master/scripts"

# Create a directory for updated tools
mkdir -p updated_p3_tools
cd updated_p3_tools

echo "📥 Downloading key p3-tools scripts..."

# Data retrieval tools (p3-all-* and p3-get-*)
echo "  📊 Data retrieval tools..."
curl -s -o p3-all-genomes.pl "$BASE_URL/p3-all-genomes.pl"
curl -s -o p3-all-contigs.pl "$BASE_URL/p3-all-contigs.pl"
curl -s -o p3-all-drugs.pl "$BASE_URL/p3-all-drugs.pl"
curl -s -o p3-all-subsystem-roles.pl "$BASE_URL/p3-all-subsystem-roles.pl"
curl -s -o p3-all-subsystems.pl "$BASE_URL/p3-all-subsystems.pl"
curl -s -o p3-all-taxonomies.pl "$BASE_URL/p3-all-taxonomies.pl"

curl -s -o p3-get-genome-data.pl "$BASE_URL/p3-get-genome-data.pl"
curl -s -o p3-get-genome-contigs.pl "$BASE_URL/p3-get-genome-contigs.pl"
curl -s -o p3-get-genome-features.pl "$BASE_URL/p3-get-genome-features.pl"
curl -s -o p3-get-feature-data.pl "$BASE_URL/p3-get-feature-data.pl"
curl -s -o p3-get-feature-sequence.pl "$BASE_URL/p3-get-feature-sequence.pl"
curl -s -o p3-get-feature-subsystems.pl "$BASE_URL/p3-get-feature-subsystems.pl"

# Computational tools (p3-submit-*)
echo "  🧮 Computational tools..."
curl -s -o p3-submit-MSA.pl "$BASE_URL/p3-submit-MSA.pl"
curl -s -o p3-submit-BLAST.pl "$BASE_URL/p3-submit-BLAST.pl"
curl -s -o p3-submit-gene-tree.pl "$BASE_URL/p3-submit-gene-tree.pl"
curl -s -o p3-submit-codon-tree.pl "$BASE_URL/p3-submit-codon-tree.pl"
curl -s -o p3-submit-genome-annotation.pl "$BASE_URL/p3-submit-genome-annotation.pl"
curl -s -o p3-submit-genome-assembly.pl "$BASE_URL/p3-submit-genome-assembly.pl"
curl -s -o p3-submit-proteome-comparison.pl "$BASE_URL/p3-submit-proteome-comparison.pl"
curl -s -o p3-submit-variation-analysis.pl "$BASE_URL/p3-submit-variation-analysis.pl"
curl -s -o p3-submit-rnaseq.pl "$BASE_URL/p3-submit-rnaseq.pl"

# Utility tools
echo "  🔧 Utility tools..."
curl -s -o p3-collate.pl "$BASE_URL/p3-collate.pl"
curl -s -o p3-count.pl "$BASE_URL/p3-count.pl"

echo "✅ Download complete!"

# Check which files were successfully downloaded
echo ""
echo "📋 Downloaded files:"
for file in *.pl; do
    if [ -s "$file" ]; then
        size=$(wc -l < "$file")
        echo "  ✅ $file ($size lines)"
    else
        echo "  ❌ $file (failed or empty)"
    fi
done

echo ""
echo "💡 Next steps:"
echo "   1. Check the parameter documentation in each .pl file"
echo "   2. To actually use these tools, copy them to your BV-BRC installation:"
echo "      sudo cp *.pl /Applications/BV-BRC.app/deployment/plbin/"
echo "   3. Update the MCP server functions to match the actual parameters"
echo "   4. Regenerate the tool reference with: python generate_tool_docs.py"
echo ""
echo "❓ Would you like to update your BV-BRC installation now? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "🔄 Updating BV-BRC installation..."
    sudo cp *.pl /Applications/BV-BRC.app/deployment/plbin/
    if [ $? -eq 0 ]; then
        echo "✅ BV-BRC installation updated successfully!"
    else
        echo "❌ Failed to update BV-BRC installation"
    fi
else
    echo "📁 Files remain in updated_p3_tools/ directory for manual inspection"
fi

cd .. 