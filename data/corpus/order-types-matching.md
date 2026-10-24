# Order types and matching on electronic markets

## Limit and market orders
A limit order specifies a maximum buying price or a minimum selling price; if it cannot be executed immediately it rests in the order book. A market order executes immediately against the best available prices in the book, accepting price uncertainty in exchange for speed of execution.

## Time-in-force instructions
An immediate-or-cancel (IOC) order executes whatever quantity is available immediately at the limit price or better, and the unexecuted remainder is cancelled. A fill-or-kill (FOK) order must be executed in full immediately, otherwise it is cancelled entirely. A good-till-cancelled order remains in the book until it is executed or cancelled.

## Other order types
A stop order becomes a market order when the market reaches a trigger price; it is commonly used to limit losses. An iceberg order displays only a small part of its total quantity in the order book, the hidden part being revealed progressively as the visible part is executed.

## Price-time priority
Most continuous markets match orders using price-time priority: the best price is executed first, and among orders at the same price, the order that arrived first is executed first. Modifying an order to increase its quantity or change its price usually makes it lose its time priority, while reducing its quantity keeps it.

## The order book
The order book lists resting buy orders (bids) and sell orders (asks) by price level. The difference between the best ask and the best bid is the bid-ask spread, an indicator of liquidity and transaction cost.
