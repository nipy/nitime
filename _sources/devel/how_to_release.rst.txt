.. _how-to-release:

=================
Releasing Nitime
=================

This section contains notes about the process that is used to release Nitime.

Most of the release process is automated by the :file:`release` script in the
:file:`tools` directory.  This is just a handy reminder for the release manager.

#. Write release notes in :file:`doc/whatsnew/` for the current release. Use
   the notes that have hopefully accumulated in
   :file:`doc/whatsnew/development`. For writing release notes, this will
   cleanly show who contributed as author of commits (get the previous release
   name from the tag list with ``git tag``)::

   git log --pretty=format:"* %aN" PREV_RELEASE... | sort | uniq

#. Uncomment the empty ``version_extra`` assignment in the :file:`version.py`
   file, so that the complete version-string will not have the ``dev`` suffix. 

#. Update the website with announcements and links to the updated files on
   github. Remember to put a short note both on the news page of the site, and
   the index ``What's New`` section.

#. Make sure that the released version of the docs is live on the site. 

#. Run :file:`build_release`, which does all the file checking and building
   that the real release script will do.  This will let you do test
   installations, check that the build procedure runs OK, etc.

#. Make the test installation **from one of the release tarballs**, make sure
   that:

   - The installation is being done into a place in your `PYTHONPATH`, which
     will over-ride your development tree. 

   - The docs build in your test installation.

   - The tests run and pass in your test installation.
  
#. Run the :file:`release` script, which makes the tar.gz, eggs and Win32 .exe
   installer. It posts them to the site and registers the release with PyPI.

#. Tag the current state of the repository::

   git tag -a rel/x.y -m"Releasing version x.y"
   git push --tags origin master

#. Draft a short release announcement with highlights of the release (and send
   it off!). 

#. Increment the version number in the :file:`version.py` file and comment the
   line with the additional ``version_extra``, so that you get back the ``dev``
   tag on the version number.

#. Commit this as the beginning of the development of the next version. 

#. Celebrate!
