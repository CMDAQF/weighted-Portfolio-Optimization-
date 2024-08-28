# weighted-Portfolio-Optimization-

This program gives you insight on how your portfolio performed against several benchmarks of your choosing, as well as optimized portfolios over a given time. Through the portfolio optimization, you can see if your allocation could have performed better, if changes to the portfolio contributed to the return, and if allocation adjustments made to the portfolio when anticipating certain market events or trends were successful.

This program takes 3 dictionaries: asset_classes, fund_allocation, and current portfolio. Current_portfolio contains your asset classes, each with your current symbols and allocation weights.  Fund_allocation contains your assets classes with your tickers and desired weight ranges to generate within. The dictionary called asset_class which provides weights for certain asset classes in "funds_allocation" in order to constrain the portfolio optimization you can also allow the assets classes to optimize within a desired range e.g.(0.2-0.34) without having the total allocation go over 100%.  
You can adjust time, benchmarks, and the number of generations.

