#!/usr/bin/env python3
"""
05_real_world_example.py - E-Commerce Assistant with Function Calling

This example demonstrates a practical, real-world e-commerce assistant
that can help customers browse products, manage their cart, and complete
purchases using function calling.

Key concepts:
- Building a stateful assistant with session management
- Combining multiple functions for complex workflows
- Error handling and validation
- Natural conversation flow with function results
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


# Simulated database
@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    stock: int
    description: str


@dataclass
class CartItem:
    product_id: str
    quantity: int
    price: float


@dataclass
class UserSession:
    user_id: str
    cart: list = field(default_factory=list)
    orders: list = field(default_factory=list)


# Mock product database
PRODUCTS = {
    "LAPTOP-001": Product(
        "LAPTOP-001",
        "ProBook 15 Laptop",
        "Electronics",
        999.99,
        15,
        "15-inch laptop with 16GB RAM, 512GB SSD",
    ),
    "LAPTOP-002": Product(
        "LAPTOP-002",
        "UltraBook Air",
        "Electronics",
        1299.99,
        8,
        "Ultra-thin laptop, 13-inch, perfect for travel",
    ),
    "PHONE-001": Product(
        "PHONE-001",
        "SmartPhone X",
        "Electronics",
        799.99,
        25,
        "Latest smartphone with advanced camera",
    ),
    "HEADPHONES-001": Product(
        "HEADPHONES-001",
        "AudioMax Pro Headphones",
        "Electronics",
        249.99,
        50,
        "Noise-canceling wireless headphones",
    ),
    "MOUSE-001": Product(
        "MOUSE-001",
        "ErgoMouse Wireless",
        "Accessories",
        49.99,
        100,
        "Ergonomic wireless mouse",
    ),
    "KEYBOARD-001": Product(
        "KEYBOARD-001",
        "MechKey Pro Keyboard",
        "Accessories",
        129.99,
        30,
        "Mechanical keyboard with RGB lighting",
    ),
    "MONITOR-001": Product(
        "MONITOR-001",
        "UltraWide 34 Monitor",
        "Electronics",
        549.99,
        12,
        "34-inch ultrawide curved monitor",
    ),
    "CHARGER-001": Product(
        "CHARGER-001",
        "FastCharge USB-C",
        "Accessories",
        29.99,
        200,
        "65W USB-C fast charger",
    ),
}

# Active user session (in a real app, this would be per-user)
current_session = UserSession(user_id="user_123")


# Tool definitions for e-commerce assistant
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products by keyword, category, or price range",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or product name",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["Electronics", "Accessories", "All"],
                        "description": "Product category filter",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price filter",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price filter",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a product to the shopping cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to add",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity to add",
                        "default": 1,
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": "Remove a product from the shopping cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to remove",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "View the current shopping cart contents",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkout",
            "description": "Process checkout and create an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipping_address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "state": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["street", "city", "state", "zip"],
                    },
                    "payment_method": {
                        "type": "string",
                        "enum": ["credit_card", "paypal", "apple_pay"],
                    },
                },
                "required": ["shipping_address", "payment_method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Check the status of an existing order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to check",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_coupon",
            "description": "Apply a coupon code to the cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "coupon_code": {
                        "type": "string",
                        "description": "The coupon code to apply",
                    },
                },
                "required": ["coupon_code"],
            },
        },
    },
]


# Function implementations
def search_products(
    query: str = None,
    category: str = "All",
    max_price: float = None,
    min_price: float = None,
) -> dict:
    """Search products based on criteria."""
    results = []

    for product in PRODUCTS.values():
        # Category filter
        if category != "All" and product.category != category:
            continue

        # Price filters
        if max_price is not None and product.price > max_price:
            continue
        if min_price is not None and product.price < min_price:
            continue

        # Keyword search
        if query:
            query_lower = query.lower()
            if (
                query_lower not in product.name.lower()
                and query_lower not in product.description.lower()
                and query_lower not in product.category.lower()
            ):
                continue

        results.append(
            {
                "id": product.id,
                "name": product.name,
                "price": product.price,
                "category": product.category,
                "in_stock": product.stock > 0,
            }
        )

    return {
        "count": len(results),
        "products": results,
    }


def get_product_details(product_id: str) -> dict:
    """Get detailed product information."""
    product = PRODUCTS.get(product_id)
    if not product:
        return {"error": f"Product {product_id} not found"}

    return {
        "id": product.id,
        "name": product.name,
        "category": product.category,
        "price": product.price,
        "description": product.description,
        "stock": product.stock,
        "in_stock": product.stock > 0,
    }


def add_to_cart(product_id: str, quantity: int = 1) -> dict:
    """Add a product to the cart."""
    product = PRODUCTS.get(product_id)
    if not product:
        return {"error": f"Product {product_id} not found"}

    if product.stock < quantity:
        return {"error": f"Only {product.stock} items available"}

    # Check if already in cart
    for item in current_session.cart:
        if item.product_id == product_id:
            item.quantity += quantity
            return {
                "success": True,
                "message": f"Updated {product.name} quantity to {item.quantity}",
                "cart_total": sum(i.price * i.quantity for i in current_session.cart),
            }

    # Add new item
    current_session.cart.append(CartItem(product_id, quantity, product.price))
    return {
        "success": True,
        "message": f"Added {quantity}x {product.name} to cart",
        "cart_total": sum(i.price * i.quantity for i in current_session.cart),
    }


def remove_from_cart(product_id: str) -> dict:
    """Remove a product from the cart."""
    for i, item in enumerate(current_session.cart):
        if item.product_id == product_id:
            product = PRODUCTS.get(product_id)
            removed = current_session.cart.pop(i)
            return {
                "success": True,
                "message": f"Removed {product.name} from cart",
                "cart_total": sum(i.price * i.quantity for i in current_session.cart),
            }

    return {"error": f"Product {product_id} not in cart"}


def view_cart() -> dict:
    """View the current cart contents."""
    if not current_session.cart:
        return {"items": [], "total": 0, "item_count": 0}

    items = []
    for cart_item in current_session.cart:
        product = PRODUCTS.get(cart_item.product_id)
        items.append(
            {
                "product_id": cart_item.product_id,
                "name": product.name if product else "Unknown",
                "quantity": cart_item.quantity,
                "unit_price": cart_item.price,
                "subtotal": cart_item.price * cart_item.quantity,
            }
        )

    total = sum(item["subtotal"] for item in items)
    return {
        "items": items,
        "item_count": sum(item["quantity"] for item in items),
        "total": total,
    }


def checkout(shipping_address: dict, payment_method: str) -> dict:
    """Process checkout."""
    if not current_session.cart:
        return {"error": "Cart is empty"}

    cart = view_cart()
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    order = {
        "order_id": order_id,
        "items": cart["items"],
        "total": cart["total"],
        "shipping": shipping_address,
        "payment": payment_method,
        "status": "confirmed",
        "estimated_delivery": "3-5 business days",
    }

    current_session.orders.append(order)
    current_session.cart = []  # Clear cart

    return {
        "success": True,
        "order_id": order_id,
        "total_charged": cart["total"],
        "status": "confirmed",
        "estimated_delivery": "3-5 business days",
        "message": "Thank you for your order!",
    }


def get_order_status(order_id: str) -> dict:
    """Get order status."""
    for order in current_session.orders:
        if order["order_id"] == order_id:
            return {
                "order_id": order_id,
                "status": order["status"],
                "items_count": len(order["items"]),
                "total": order["total"],
                "estimated_delivery": order["estimated_delivery"],
            }

    return {"error": f"Order {order_id} not found"}


def apply_coupon(coupon_code: str) -> dict:
    """Apply a coupon code."""
    valid_coupons = {
        "SAVE10": {"discount": 10, "type": "percent"},
        "SAVE20": {"discount": 20, "type": "percent"},
        "FLAT50": {"discount": 50, "type": "fixed"},
    }

    coupon = valid_coupons.get(coupon_code.upper())
    if not coupon:
        return {"error": "Invalid coupon code"}

    cart = view_cart()
    if cart["total"] == 0:
        return {"error": "Cart is empty"}

    if coupon["type"] == "percent":
        discount = cart["total"] * (coupon["discount"] / 100)
    else:
        discount = coupon["discount"]

    return {
        "success": True,
        "coupon": coupon_code.upper(),
        "discount_amount": round(discount, 2),
        "new_total": round(cart["total"] - discount, 2),
    }


FUNCTION_MAP = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "add_to_cart": add_to_cart,
    "remove_from_cart": remove_from_cart,
    "view_cart": view_cart,
    "checkout": checkout,
    "get_order_status": get_order_status,
    "apply_coupon": apply_coupon,
}


def execute_function(function_name: str, arguments: dict) -> str:
    """Execute a function and return JSON result."""
    if function_name in FUNCTION_MAP:
        try:
            result = FUNCTION_MAP[function_name](**arguments)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"Unknown function: {function_name}"})


class ECommerceAssistant:
    """E-commerce shopping assistant powered by function calling."""

    def __init__(self):
        self.conversation_history = []
        self.system_prompt = """You are a helpful e-commerce shopping assistant for TechStore.

You help customers:
- Find products they're looking for
- Get product details and recommendations
- Manage their shopping cart
- Complete purchases
- Track orders

Be friendly, helpful, and proactive in suggesting relevant products.
When showing search results, format them nicely for the customer.
Always confirm important actions like checkout.

Available products: laptops, phones, headphones, monitors, keyboards, mice, and chargers."""

        self.conversation_history.append({"role": "system", "content": self.system_prompt})

    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's response."""
        self.conversation_history.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=self.conversation_history,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        while assistant_message.tool_calls:
            self.conversation_history.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                console.print(f"  [dim]Executing: {function_name}({json.dumps(arguments)})[/dim]")

                result = execute_function(function_name, arguments)

                self.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            response = client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=self.conversation_history,
                tools=TOOLS,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message

        self.conversation_history.append(assistant_message)
        return assistant_message.content


def display_available_products():
    """Display all available products in a table."""
    table = Table(title="TechStore Product Catalog")
    table.add_column("ID", style="cyan")
    table.add_column("Product", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Price", style="magenta", justify="right")
    table.add_column("Stock", justify="right")

    for product in PRODUCTS.values():
        table.add_row(
            product.id,
            product.name,
            product.category,
            f"${product.price:.2f}",
            str(product.stock),
        )

    console.print(table)


def main():
    console.print(
        Panel.fit(
            "[bold blue]E-Commerce Shopping Assistant[/bold blue]\n"
            "A practical function calling example",
            border_style="blue",
        )
    )

    # Show available products
    console.print("\n[bold yellow]Available Products:[/bold yellow]")
    display_available_products()

    # Create assistant
    assistant = ECommerceAssistant()

    # Simulate a shopping conversation
    conversation = [
        "Hi! I'm looking for a laptop under $1100",
        "Tell me more about the ProBook 15",
        "Add it to my cart",
        "I also need wireless headphones, what do you have?",
        "Add the AudioMax Pro headphones too",
        "What's in my cart?",
        "I have a coupon code SAVE10",
        "I'm ready to checkout. Ship to 123 Tech Street, San Francisco, CA 94102. Pay with credit card.",
    ]

    console.print("\n[bold yellow]Shopping Session:[/bold yellow]")
    console.print("[dim]Simulating a customer shopping experience...[/dim]\n")

    for message in conversation:
        console.print(f"\n[bold green]Customer:[/bold green] {message}")
        response = assistant.chat(message)
        console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")
        console.print("-" * 60)

    # Show final order
    if current_session.orders:
        console.print("\n[bold yellow]Order Summary:[/bold yellow]")
        for order in current_session.orders:
            console.print(f"  Order ID: {order['order_id']}")
            console.print(f"  Status: {order['status']}")
            console.print(f"  Total: ${order['total']:.2f}")
            console.print(f"  Delivery: {order['estimated_delivery']}")


if __name__ == "__main__":
    main()
