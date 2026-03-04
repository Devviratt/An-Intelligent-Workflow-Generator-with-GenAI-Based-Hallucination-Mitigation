#!/usr/bin/env python
import asyncio
from src.engines.llm_workflow_generator import LLMWorkflowGenerator
from src.engines.domain_engine import DomainDatasetEngine
from src.engines.instruction_parser import InstructionParser

async def test_rag():
    gen = LLMWorkflowGenerator()
    dataset_engine = DomainDatasetEngine()
    parser = InstructionParser()
    
    await dataset_engine.load_all()
    datasets = dataset_engine.all_datasets()
    parser.fit(datasets)
    
    # Get online_payment dataset
    dataset = dataset_engine.get('online_payment')
    
    # Parse actual instruction
    parsed = parser.parse("Generate a payment processing workflow with fraud detection")
    
    # Test the context building
    context = gen._build_concise_context(dataset, parsed)
    print("RAG Context:")
    print("=" * 60)
    print(context)
    print("=" * 60)
    print(f"\nContext size: {len(context)} characters")
    
    # Test generation
    try:
        print("\nGenerating workflow with RAG...")
        workflow = await gen.generate(dataset, parsed)
        print(f"✓ Workflow generated!")
        print(f"  Nodes: {len(workflow.processes)}")
        print(f"  Edges: {len(workflow.transitions)}")
        if workflow.processes:
            print(f"  First node: {workflow.processes[0].id}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag())

