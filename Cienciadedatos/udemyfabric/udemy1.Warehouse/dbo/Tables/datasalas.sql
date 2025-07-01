CREATE TABLE [dbo].[datasalas] (

	[AddressID] int NULL, 
	[AddressLine1] varchar(8000) NULL, 
	[AddressLine2] varchar(8000) NULL, 
	[City] varchar(8000) NULL, 
	[StateProvince] varchar(8000) NULL, 
	[CountryRegion] varchar(8000) NULL, 
	[PostalCode] varchar(8000) NULL, 
	[rowguid] uniqueidentifier NULL, 
	[ModifiedDate] datetime2(3) NULL
);