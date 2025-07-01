CREATE TABLE [dbo].[DimProductSubcategory] (

	[ProductSubcategoryKey] bigint NULL, 
	[ProductSubcategoryAlternateKey] bigint NULL, 
	[EnglishProductSubcategoryName] varchar(8000) NULL, 
	[SpanishProductSubcategoryName] varchar(8000) NULL, 
	[FrenchProductSubcategoryName] varchar(8000) NULL, 
	[ProductCategoryKey] bigint NULL
);