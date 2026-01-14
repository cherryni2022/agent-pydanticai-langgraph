"""
Send API 高级示例 - 展示如何处理复杂的动态并行场景

场景: planAgent 生成不同类型的 subtask (SQL查询、API调用、数据处理)
每种类型需要不同的 worker 处理
"""

import asyncio
from typing import TypedDict, List, Annotated, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from langgraph.graph import StateGraph, END
from langgraph.types import Send
import operator
from datetime import datetime

# ============= Pydantic Models =============

class SubTask(BaseModel):
    """子任务定义"""
    task_id: int
    task_type: Literal["sql", "api", "compute"]  # 不同类型的任务
    description: str
    priority: int = 1
    estimated_time: float = 1.0  # 预估执行时间(秒)

class PlanResult(BaseModel):
    """Plan Agent 输出"""
    original_query: str
    subtasks: List[SubTask] = Field(description="不同类型的子任务")
    execution_strategy: str = "parallel"

class TaskResult(BaseModel):
    """任务执行结果"""
    task_id: int
    task_type: str
    result: str
    execution_time: float
    success: bool = True

# ============= State Definition =============

class AdvancedState(TypedDict):
    """高级状态定义"""
    query: str
    plan: PlanResult | None
    task_results: Annotated[List[TaskResult], operator.add]
    final_answer: str
    execution_stats: dict

# ============= PydanticAI Agents =============

plan_agent = Agent(
    'openai:gpt-4',
    result_type=PlanResult,
    system_prompt="""你是任务规划专家。
    将复杂查询分解为不同类型的子任务:
    - sql: 数据库查询任务
    - api: 外部 API 调用任务
    - compute: 数据计算任务
    
    根据任务复杂度设置优先级和预估时间。
    """
)

sql_agent = Agent(
    'openai:gpt-4',
    result_type=str,
    system_prompt="你是 SQL 专家。生成准确的 SQL 查询。"
)

api_agent = Agent(
    'openai:gpt-4',
    result_type=str,
    system_prompt="你是 API 集成专家。生成 API 调用代码。"
)

compute_agent = Agent(
    'openai:gpt-4',
    result_type=str,
    system_prompt="你是数据分析专家。生成数据处理代码。"
)

# ============= Graph Nodes =============

async def plan_node(state: AdvancedState) -> dict:
    """规划节点"""
    print(f"\n{'='*80}")
    print(f"📋 [Plan Agent] 分析查询: {state['query'][:50]}...")
    print(f"{'='*80}")
    
    result = await plan_agent.run(state['query'])
    plan = result.data
    
    print(f"\n✅ 生成 {len(plan.subtasks)} 个子任务:")
    
    # 按类型统计
    type_counts = {}
    for task in plan.subtasks:
        type_counts[task.task_type] = type_counts.get(task.task_type, 0) + 1
        print(f"  [{task.task_type.upper():7}] Task {task.task_id} (P{task.priority}, ~{task.estimated_time}s): {task.description[:60]}")
    
    print(f"\n📊 任务类型分布: {type_counts}")
    
    return {
        "plan": plan,
        "task_results": [],
        "execution_stats": {
            "total_tasks": len(plan.subtasks),
            "type_distribution": type_counts,
            "start_time": datetime.now().isoformat()
        }
    }

def route_to_workers(state: AdvancedState) -> List[Send]:
    """
    智能路由: 根据任务类型分发到不同的 worker
    这是 Send API 的核心优势 - 可以动态选择目标节点
    """
    plan = state.get("plan")
    if not plan or not plan.subtasks:
        return []
    
    sends = []
    
    print(f"\n{'='*80}")
    print(f"🔀 [Router] 智能分发 {len(plan.subtasks)} 个任务")
    print(f"{'='*80}")
    
    for task in plan.subtasks:
        # 根据任务类型选择不同的 worker 节点
        worker_node = f"{task.task_type}_worker"
        
        sends.append(
            Send(
                worker_node,
                {
                    "subtask": task,
                    "start_time": datetime.now().isoformat()
                }
            )
        )
        
        print(f"  ➡️  Task {task.task_id} ({task.task_type}) -> {worker_node}")
    
    return sends

async def sql_worker(state: dict) -> dict:
    """SQL Worker - 处理 SQL 任务"""
    subtask: SubTask = state["subtask"]
    start_time = datetime.fromisoformat(state["start_time"])
    
    print(f"\n  🗄️  [SQL Worker] 执行 Task {subtask.task_id}...")
    
    # 模拟执行
    await asyncio.sleep(min(subtask.estimated_time, 0.5))  # 模拟延迟
    
    result = await sql_agent.run(subtask.description)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  ✅ [SQL Worker] Task {subtask.task_id} 完成 ({execution_time:.2f}s)")
    
    return {
        "task_results": [TaskResult(
            task_id=subtask.task_id,
            task_type="sql",
            result=result.data,
            execution_time=execution_time
        )]
    }

async def api_worker(state: dict) -> dict:
    """API Worker - 处理 API 调用任务"""
    subtask: SubTask = state["subtask"]
    start_time = datetime.fromisoformat(state["start_time"])
    
    print(f"\n  🌐 [API Worker] 执行 Task {subtask.task_id}...")
    
    await asyncio.sleep(min(subtask.estimated_time, 0.5))
    
    result = await api_agent.run(subtask.description)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  ✅ [API Worker] Task {subtask.task_id} 完成 ({execution_time:.2f}s)")
    
    return {
        "task_results": [TaskResult(
            task_id=subtask.task_id,
            task_type="api",
            result=result.data,
            execution_time=execution_time
        )]
    }

async def compute_worker(state: dict) -> dict:
    """Compute Worker - 处理计算任务"""
    subtask: SubTask = state["subtask"]
    start_time = datetime.fromisoformat(state["start_time"])
    
    print(f"\n  🧮 [Compute Worker] 执行 Task {subtask.task_id}...")
    
    await asyncio.sleep(min(subtask.estimated_time, 0.5))
    
    result = await compute_agent.run(subtask.description)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  ✅ [Compute Worker] Task {subtask.task_id} 完成 ({execution_time:.2f}s)")
    
    return {
        "task_results": [TaskResult(
            task_id=subtask.task_id,
            task_type="compute",
            result=result.data,
            execution_time=execution_time
        )]
    }

async def aggregate_node(state: AdvancedState) -> dict:
    """汇总节点 - 整合所有结果"""
    results = state.get("task_results", [])
    stats = state.get("execution_stats", {})
    
    print(f"\n{'='*80}")
    print(f"📊 [Aggregator] 汇总 {len(results)} 个结果")
    print(f"{'='*80}")
    
    # 按类型分组
    results_by_type = {}
    total_time = 0
    
    for result in results:
        if result.task_type not in results_by_type:
            results_by_type[result.task_type] = []
        results_by_type[result.task_type].append(result)
        total_time += result.execution_time
    
    # 生成最终答案
    final_answer = f"执行摘要:\n\n"
    final_answer += f"总任务数: {len(results)}\n"
    final_answer += f"总执行时间: {total_time:.2f}s\n"
    final_answer += f"平均执行时间: {total_time/len(results):.2f}s\n\n"
    
    for task_type, type_results in results_by_type.items():
        final_answer += f"\n{task_type.upper()} 任务 ({len(type_results)} 个):\n"
        for result in type_results:
            final_answer += f"  - Task {result.task_id}: {result.result[:80]}...\n"
    
    # 更新统计信息
    stats.update({
        "end_time": datetime.now().isoformat(),
        "total_execution_time": total_time,
        "results_by_type": {k: len(v) for k, v in results_by_type.items()}
    })
    
    print(f"\n✨ 执行统计:")
    print(f"  - 总任务: {len(results)}")
    print(f"  - 总时间: {total_time:.2f}s")
    print(f"  - 类型分布: {stats['results_by_type']}")
    
    return {
        "final_answer": final_answer,
        "execution_stats": stats
    }

# ============= Graph Construction =============

def create_advanced_graph() -> StateGraph:
    """
    创建高级图 - 使用 Send API 实现智能路由
    
    特点:
    1. 动态任务数量
    2. 不同类型任务路由到不同 worker
    3. 所有 worker 并行执行
    4. 自动汇总结果
    """
    workflow = StateGraph(AdvancedState)
    
    # 添加节点
    workflow.add_node("plan", plan_node)
    workflow.add_node("sql_worker", sql_worker)
    workflow.add_node("api_worker", api_worker)
    workflow.add_node("compute_worker", compute_worker)
    workflow.add_node("aggregate", aggregate_node)
    
    # 设置入口
    workflow.set_entry_point("plan")
    
    # 关键: 使用 conditional_edges 实现智能路由
    workflow.add_conditional_edges(
        "plan",
        route_to_workers,  # 返回 List[Send]
        # Send 会自动路由到对应的 worker
    )
    
    # 所有 worker 完成后汇总
    workflow.add_edge("sql_worker", "aggregate")
    workflow.add_edge("api_worker", "aggregate")
    workflow.add_edge("compute_worker", "aggregate")
    workflow.add_edge("aggregate", END)
    
    return workflow.compile()

# ============= Main Function =============

async def main():
    """运行高级示例"""
    print("\n" + "="*80)
    print("🚀 Send API 高级示例 - 智能路由与动态并行")
    print("="*80)
    
    # 创建图
    graph = create_advanced_graph()
    
    # 测试查询
    queries = [
        """
        分析科技股表现并生成报告:
        1. 查询 FAANG 股票的最新价格 (SQL)
        2. 调用财经 API 获取实时新闻 (API)
        3. 计算过去一年的收益率 (Compute)
        4. 查询历史交易量数据 (SQL)
        5. 调用分析 API 获取分析师评级 (API)
        """,
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'#'*80}")
        print(f"# 测试 {i}")
        print(f"{'#'*80}")
        
        initial_state = {
            "query": query,
            "plan": None,
            "task_results": [],
            "final_answer": "",
            "execution_stats": {}
        }
        
        # 执行图
        final_state = await graph.ainvoke(initial_state)
        
        print(f"\n{'='*80}")
        print("✨ 最终结果")
        print(f"{'='*80}")
        print(final_state["final_answer"])
        
        print(f"\n📈 执行统计:")
        print(f"  {final_state['execution_stats']}")
    
    # 可视化图结构
    print(f"\n{'='*80}")
    print("📊 图结构可视化")
    print(f"{'='*80}")
    
    try:
        mermaid = graph.get_graph().draw_mermaid()
        print(mermaid)
        
        # 保存为文件
        with open("send_api_graph.mmd", "w") as f:
            f.write(mermaid)
        print("\n💾 图结构已保存到 send_api_graph.mmd")
        
    except Exception as e:
        print(f"无法生成可视化: {e}")

if __name__ == "__main__":
    asyncio.run(main())
