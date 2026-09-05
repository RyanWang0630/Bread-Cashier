Bread-Cashier
Automated Bakery Cashier & Bread Detection System

An AI-powered automated point-of-sale vision node that identifies bakery items, calculates checkout totals, and speeds up transactions

//

Overview

This project is mainly encouraged by the idea of bakery checkouts. Traditional retail products use barcodes, but fresh bakery are unpackaged and lack barcodes. Cashiers have to manually recognize each item and enter it into the system, which causes long queues and slows down transaction speeds. This project aims to solve these checkout delays by automating the classification process using AI. In this project, a custom dataset of various pastry categories is gathered and trained using Roboflow. A camera  captures an image of the customer's tray, runs AI object detection to classify each distinct pastry type, and automatically calculates the final total price based on the detected items.

Main Issue: Unpackaged bakery items require manual cashier identification, leading to checkout bottlenecks, higher labor overhead, and slow customer turnover.

The Goal this project tends to aim: A low-cost, automated cashier vision station that rapidly identifies multiple bakery items on a single tray, classifies their specific categories, calculates the final checkout total, and streamlines the retail process.

//

Features

Category Classification: Classifies different pastry categories (e.g., croissants, baguettes, donuts) simultaneously using a custom model trained in Roboflow.
Automated Pricing: Matches each recognized pastry class with its pre-configured price tag and calculates the final checkout total automatically.
Multi-Item Detection: Detects multiple types of pastries rather than one classification at a time.
Real-Time POS Feedback: Transmits detected item categories, individual prices, and the total sum directly to the display interface or a website(my code)

//

Hardware

Camera: Your own camera
Software Language: C++ / Arduino Framework AI: Custom Model (YOLOv11) ; Roboflow

