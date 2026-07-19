# RecoMart Data Quality Report

Generated: 2026-07-19T03:38:11

## 1. Schema conformance

- **customers** (99441 rows, 5 cols): PASS
- **orders** (99441 rows, 8 cols): PASS
- **order_items** (112650 rows, 7 cols): PASS
- **order_reviews** (99224 rows, 7 cols): PASS
- **order_payments** (103886 rows, 5 cols): PASS
- **products** (32951 rows, 9 cols): PASS
- **sellers** (3095 rows, 4 cols): PASS
- **category_translation** (71 rows, 2 cols): PASS

## 2. Missing values (non-zero columns only)

- **customers**: no missing values
- **orders**: {'order_approved_at': 160, 'order_delivered_carrier_date': 1783, 'order_delivered_customer_date': 2965}
- **order_items**: no missing values
- **order_reviews**: {'review_comment_title': 87656, 'review_comment_message': 58247}
- **order_payments**: no missing values
- **products**: {'product_category_name': 610, 'product_name_lenght': 610, 'product_description_lenght': 610, 'product_photos_qty': 610, 'product_weight_g': 2, 'product_length_cm': 2, 'product_height_cm': 2, 'product_width_cm': 2}
- **sellers**: no missing values
- **category_translation**: no missing values

## 3. Duplicate primary keys

- **customers**: 0 duplicate `customer_id` rows
- **orders**: 0 duplicate `order_id` rows
- **order_items**: 0 duplicate `` rows
- **order_reviews**: 814 duplicate `review_id` rows
- **order_payments**: 0 duplicate `` rows
- **products**: 0 duplicate `product_id` rows
- **sellers**: 0 duplicate `seller_id` rows
- **category_translation**: 0 duplicate `` rows

## 4. Range / format checks

- **review_score_out_of_1_5_range**: 0 violating rows
- **negative_or_zero_price**: 0 violating rows
- **negative_freight_value**: 0 violating rows
- **negative_payment_value**: 0 violating rows

## 5. Referential integrity

- **order_items_orphaned_order_id**: 0 orphaned rows
- **order_reviews_orphaned_order_id**: 0 orphaned rows
- **order_payments_orphaned_order_id**: 0 orphaned rows
- **order_items_orphaned_product_id**: 0 orphaned rows
- **orders_orphaned_customer_id**: 0 orphaned rows

## Summary

Total rows checked: 550759. Total violating rows across all checks: 814. Schema conformance: PASS.
