# Retrieval Optimization Report

## 1. Problem Statement

Mục tiêu của bài toán là xây dựng một hệ thống Retrieval cho tài liệu bệnh hại lúa nhằm hỗ trợ RAG (Retrieval-Augmented Generation).

Đầu vào là các câu hỏi tự nhiên của người dùng như:

* Cách điều trị bệnh Đạo ôn.
* Dấu hiệu nhận biết bệnh Bạc lá.
* Thuốc Tervigo được sử dụng cho bệnh nào.

Đầu ra là các đoạn thông tin (chunks) liên quan nhất để cung cấp cho mô hình ngôn ngữ ở bước sinh câu trả lời.

---

## 2. Approach

### 2.1 Data Processing

Pipeline:

PDF → Text Extraction → Data Cleaning → Chunking → Retrieval

Các bước thực hiện:

* Trích xuất văn bản từ PDF bằng PyMuPDF.
* Làm sạch dữ liệu sau khi extract.
* Xây dựng dữ liệu chunk phục vụ Retrieval.

---

### 2.2 Structure-based Chunking

Ý tưởng:

1 Disease = 1 Chunk

Ưu điểm:

* Ít vector.
* Tốc độ indexing nhanh.

Nhược điểm:

* Một chunk chứa quá nhiều thông tin.
* Query cụ thể thường trả về toàn bộ bệnh thay vì nội dung cần thiết.

---

### 2.3 Section-based Chunking

Ý tưởng:

1 Disease + 1 Section = 1 Chunk

Ví dụ:

* Triệu chứng
* Nguyên nhân
* Điều trị
* Phòng ngừa

Workflow:

Disease Detection → Section Detection → Chunk Generation

Ưu điểm:

* Trả về thông tin chính xác hơn.
* Giảm lượng context dư thừa.

Nhược điểm:

* Số lượng vector tăng lên.

---

## 3. Retrieval Methods

### 3.1 BM25

Workflow:

Document → Tokenization → BM25 Index → Ranking

Đặc điểm:

* Dựa trên từ khóa.
* Hoạt động tốt với tên bệnh và tên thuốc.

---

### 3.2 Cosine Similarity

Workflow:

Query → Embedding → FAISS → Cosine Similarity

Đặc điểm:

* Tìm kiếm theo ngữ nghĩa.
* Hiểu được các câu diễn đạt khác nhau.

---

### 3.3 L2 Distance

Workflow:

Query → Embedding → FAISS → L2 Distance

Đặc điểm:

* Đo khoảng cách giữa các vector.
* Khoảng cách càng nhỏ thì độ tương đồng càng cao.

---

### 3.4 Hybrid Search

Công thức:

Hybrid Score = α × BM25 + (1 − α) × Semantic Score

Mục tiêu:

* Kết hợp khả năng hiểu ngữ nghĩa của Semantic Search.
* Kết hợp khả năng bắt từ khóa của BM25.

---

## 4. Evaluation Result

Dataset:

* 20 test cases.

| Method | Top-1 | Top-3 |
| ------ | ----- | ----- |
| BM25   | 75%   | 90%   |
| Cosine | 55%   | 85%   |
| L2     | 60%   | 75%   |
| Hybrid | 85%   | 90%   |

Kết luận:

* BM25 hoạt động tốt với dữ liệu chứa nhiều thuật ngữ chuyên ngành.
* Semantic Search đơn lẻ chưa đạt hiệu quả cao.
* Hybrid Search cho kết quả tốt nhất.

---

## 5. Difficulties

### 5.1 Fixed-size Chunking

Vấn đề:

* Nội dung bị cắt giữa chừng.
* Mất ngữ cảnh.

Nguyên nhân:

* Chunk được tạo theo số lượng ký tự thay vì cấu trúc tài liệu.

Giải pháp:

* Chuyển sang Structure-based Chunking.
* Chuyển sang Section-based Chunking.

---

### 5.2 PDF Structure Loss

Vấn đề:

* Sau khi extract, một số cấu trúc trong PDF bị thay đổi.

Hệ quả:

* Khó xác định ranh giới giữa các mục.

Giải pháp:

* Data Cleaning.
* Rule-based Detection.

---

### 5.3 Retrieval Error

Ví dụ:

Query:

"Dấu hiệu nhận biết bệnh Bạc lá"

Kết quả trả về chưa chính xác.

Nguyên nhân:

* Tokenization đơn giản bằng split().
* Chưa xử lý tốt cụm từ tiếng Việt.

Giải pháp đề xuất:

* Vietnamese Tokenization.

---

### 5.4 Synonym Problem

Ví dụ:

* xử lý
* điều trị

Hai cụm từ có nghĩa gần giống nhau nhưng BM25 xem là khác nhau.

Giải pháp đề xuất:

* Query Expansion.
* Semantic Search.

---

## 6. Current Status

Completed:

* PDF Extraction
* Data Cleaning
* Structure-based Chunking
* Section-based Chunking
* BM25
* Cosine Similarity
* L2 Distance
* Hybrid Search
* Evaluation
* Error Analysis

---

## 7. Future Work

* Vietnamese Tokenization.
* Alpha Tuning cho Hybrid Search.
* Query Expansion.
* Embedding Model Comparison.
* Tối ưu Retrieval Pipeline.
