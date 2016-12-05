[numthreads(1, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
}

//
//public ref class Watcher
//{
//private:
//
//	// Define the event handlers.
//	static void OnChanged(Object^ source, FileSystemEventArgs^ e)
//	{
//
//		//convert filepath from managed string to standard string before passing to the file opener
//		msclr::interop::marshal_context context;
//		std::string stdFullPath = context.marshal_as<std::string>(e->FullPath);
//
//		string STRING;
//		ifstream infile;
//		infile.open(stdFullPath);
//		while (!infile.eof()) // To get you all the lines.
//		{
//			getline(infile, STRING); // Saves the line in STRING.
//			cout << STRING; // Prints our STRING.
//		}
//		infile.close();
//		// Specify what is done when a file is changed, created, or deleted.
//		//Console::WriteLine("File: {0} {1}", e->FullPath, e->ChangeType);
//	}
//
//	static void OnRenamed(Object^ /*source*/, RenamedEventArgs^ e)
//	{
//		// Specify what is done when a file is renamed.
//		Console::WriteLine("File: {0} renamed to {1}", e->OldFullPath, e->FullPath);
//	}
//
//public:
//
//	[PermissionSet(SecurityAction::Demand, Name = "FullTrust")]
//	void static FileWatch()
//	{
//		string path = __FILE__; //gets source code path, include file name
//		path = path.substr(0, 1 + path.find_last_of('\\')); //removes file name
//		path += ("test.txt"); //adds input file to path
//		path = "\\" + path;
//		String^ p = gcnew String(path.c_str());
//		// Create a new FileSystemWatcher and set its properties.
//		FileSystemWatcher^ watcher = gcnew FileSystemWatcher;
//		watcher->Path = "C:\\rnnRead\\";
//
//		/* Watch for changes in LastAccess and LastWrite times, and
//		the renaming of files or directories. */
//		watcher->NotifyFilter = static_cast<NotifyFilters>(NotifyFilters::LastAccess |
//			NotifyFilters::LastWrite | NotifyFilters::FileName | NotifyFilters::DirectoryName);
//
//		// Only watch text files.
//		watcher->Filter = "*.txt";
//
//		// Add event handlers.
//		watcher->Changed += gcnew FileSystemEventHandler(Watcher::OnChanged);
//		//watcher->Created += gcnew FileSystemEventHandler(Watcher::OnChanged);
//		//watcher->Deleted += gcnew FileSystemEventHandler(Watcher::OnChanged);
//		//watcher->Renamed += gcnew RenamedEventHandler(Watcher::OnRenamed);
//
//		// Begin watching.
//		watcher->EnableRaisingEvents = true;
//
//		// Wait for the user to quit the program.
//		Console::WriteLine("Press \'q\' to quit the sample.");
//		while (1>0)
//		{
//		}
//		getchar();
//	}
//};
