# LangChain vs LangGraph


## Key differences

> https://www.designveloper.com/blog/langgraph-vs-langchain-comparison/

|Aspect|LangChain|LangGraph|
|------|---------|---------|
|Workflow Structure | Linear chain (or DAG) where steps run in a defined sequence.|Graph of nodes and edges allowing loops and branching for dynamic flows.|
|Design Patterns|Code-driven chains; developers write Python scripts to define each step (imperative logic).|Graph-based, declarative workflows; tasks are configured as connected nodes, often via a visual interface.|
|State Management|Limited persistence – data can pass from step to step, but no built-in long-term state across runs without custom handling.|Robust shared state – a central state object that all nodes read/write, enabling persistent memory and context throughout the session.|
|Flexibility and Control|Highly flexible customization via code; any logic can be implemented, but flow control (loops, retries) must be coded manually.|Rich built-in control flow primitives (conditionals, loops, retries, wait states) for complex logic without extra code. The structured approach adds control but with some constraints on low-level tweaks.|
|Code Complexity & Maintainability|Straightforward for simple tasks, but becomes complex as logic grows – long chains can be hard to debug and maintain.|Handles complex workflows with less code by organizing logic in a clear graph; easier to visualize, trace, and maintain large agent systems.|
|Proxy Implementation|No built-in web scraping or proxy support – relies on external HTTP clients or tools for web access. Proxies can be configured at the network request level when needed.|Similar approach – supports proxies for agents that call external sites, but requires configuring the underlying requests or using proxy-integrated tools. No automatic proxy handling is included by default.|

